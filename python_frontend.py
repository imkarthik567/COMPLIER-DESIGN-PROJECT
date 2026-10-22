"""Lower a documented integer-focused Python subset into the project's three-address IR."""
import ast

from ir import Assign, BinOp, Call, Function, Goto, IfZ, Label, ParallelSum, ParallelReduce, Print, Return, IRError

PARALLEL_REDUCTIONS = (
    "parallel_sum", "parallel_min", "parallel_max", "parallel_product",
    "parallel_count", "parallel_any", "parallel_all", "parallel_bit_and",
    "parallel_bit_or", "parallel_bit_xor", "parallel_gcd", "parallel_lcm",
    "parallel_average", "parallel_argmin", "parallel_argmax",
)


class PythonSubsetError(IRError):
    """Python syntax outside the source subset supported by this compiler."""


class _ReadNames(ast.NodeVisitor):
    def __init__(self):
        self.names = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)

    def visit_Call(self, node):
        # A direct function name is a symbol, not a local variable read.
        if node.func is not None and not isinstance(node.func, ast.Name):
            self.visit(node.func)
        args = node.args[1:] if isinstance(node.func, ast.Name) and node.func.id in PARALLEL_REDUCTIONS else node.args
        for arg in args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)


def _check_reads(expr, assigned):
    reads = _ReadNames()
    reads.visit(expr)
    missing = sorted(reads.names - assigned)
    if missing:
        raise PythonSubsetError(f"line {getattr(expr, 'lineno', 0)}: variable(s) read before assignment: {', '.join(missing)}")


def _validate_assignments(statements, assigned):
    """Reject Python local reads that the VM's zero-filled slots would otherwise hide."""
    current = set(assigned)
    for node in statements:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            _check_reads(node.value, current)
            current.add(node.targets[0].id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            _check_reads(node.value, current)
            current.add(node.target.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            _check_reads(node.target, current)
            _check_reads(node.value, current)
            current.add(node.target.id)
        elif isinstance(node, ast.Expr):
            _check_reads(node.value, current)
        elif isinstance(node, ast.Return) and node.value:
            _check_reads(node.value, current)
        elif isinstance(node, ast.If):
            _check_reads(node.test, current)
            yes = _validate_assignments(node.body, current)
            no = _validate_assignments(node.orelse, current) if node.orelse else set(current)
            current = yes & no
        elif isinstance(node, ast.While):
            _check_reads(node.test, current)
            _validate_assignments(node.body, current)
        elif isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
            for arg in node.iter.args:
                _check_reads(arg, current)
            body_state = set(current)
            if isinstance(node.target, ast.Name):
                body_state.add(node.target.id)
            _validate_assignments(node.body, body_state)
    return current


def _always_returns(statements):
    for node in statements:
        if isinstance(node, ast.Return):
            return True
        if isinstance(node, ast.If) and node.orelse and _always_returns(node.body) and _always_returns(node.orelse):
            return True
    return False


class _Lowerer:
    OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.FloorDiv: "//", ast.Mod: "%%",
           ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}

    def __init__(self, name, params, reserved=()):
        self.fn = Function(name, params, [])
        self.temp_id = 0
        self.label_id = 0
        self.reserved = set(reserved) | set(params) | {name}
        self.tuples = {}
        self.control_depth = 0

    def emit(self, node, lineno):
        node.line = lineno
        self.fn.body.append(node)

    def temp(self):
        while True:
            name = f"__cvm_tmp_{self.temp_id}"
            self.temp_id += 1
            if name not in self.reserved:
                self.reserved.add(name)
                break
        return name

    def label(self, stem):
        name = f"_{stem}{self.label_id}"
        self.label_id += 1
        return name

    def expr(self, node):
        line = getattr(node, "lineno", self.fn.line)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Constant) and type(node.value) is bool:
            return int(node.value)
        if isinstance(node, ast.Name):
            if node.id in self.tuples:
                raise PythonSubsetError(f"line {line}: tuple values must be accessed with a constant index")
            return node.id
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            values = self.tuples.get(node.value.id)
            index = node.slice
            if isinstance(index, ast.UnaryOp) and isinstance(index.op, ast.USub) and isinstance(index.operand, ast.Constant) and type(index.operand.value) is int:
                index_value = -index.operand.value
            elif isinstance(index, ast.Constant) and type(index.value) is int:
                index_value = index.value
            else:
                index_value = None
            if values is None or index_value is None:
                raise PythonSubsetError(f"line {line}: tuple indexing requires a local fixed tuple and a constant integer index")
            normalized = index_value + len(values) if index_value < 0 else index_value
            if not 0 <= normalized < len(values):
                raise PythonSubsetError(f"line {line}: tuple index out of range")
            return values[normalized]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            if isinstance(node.operand, ast.Constant) and type(node.operand.value) is int:
                return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
            value = self.expr(node.operand)
            if isinstance(node.op, ast.UAdd):
                return value
            dst = self.temp()
            self.emit(BinOp(dst, "-", 0, value), line)
            return dst
        if isinstance(node, ast.BinOp) and type(node.op) in self.OPS:
            left, right = self.expr(node.left), self.expr(node.right)
            dst = self.temp()
            self.emit(BinOp(dst, self.OPS[type(node.op)], left, right), line)
            return dst
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1 and type(node.ops[0]) in self.OPS:
            left, right = self.expr(node.left), self.expr(node.comparators[0])
            dst = self.temp()
            self.emit(BinOp(dst, self.OPS[type(node.ops[0])], left, right), line)
            return dst
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id != "print":
            if node.keywords:
                raise PythonSubsetError(f"line {line}: keyword arguments are unsupported")
            if node.func.id in PARALLEL_REDUCTIONS:
                if len(node.args) != 3 or not isinstance(node.args[0], ast.Name):
                    raise PythonSubsetError(f"line {line}: {node.func.id} expects (worker_function, start, stop)")
                start, stop = self.expr(node.args[1]), self.expr(node.args[2])
                dst = self.temp()
                if node.func.id == "parallel_sum":
                    self.emit(ParallelSum(dst, node.args[0].id, start, stop), line)
                else:
                    self.emit(ParallelReduce(dst, node.args[0].id, start, stop, node.func.id.removeprefix("parallel_")), line)
                return dst
            args = [self.expr(arg) for arg in node.args]
            dst = self.temp()
            self.emit(Call(dst, node.func.id, args), line)
            return dst
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            raise PythonSubsetError(f"line {line}: print can only be used as a statement")
        raise PythonSubsetError(f"line {line}: unsupported expression: {ast.unparse(node)}")

    def block(self, statements):
        for node in statements:
            line = getattr(node, "lineno", 0)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if isinstance(node.value, ast.Tuple):
                    name = node.targets[0].id
                    if self.control_depth:
                        raise PythonSubsetError(f"line {line}: tuple creation inside a branch or loop is unsupported")
                    if not node.value.elts:
                        raise PythonSubsetError(f"line {line}: empty tuples are unsupported")
                    elements = [self.expr(element) for element in node.value.elts]
                    slots = []
                    for element in elements:
                        slot = self.temp()
                        self.emit(Assign(slot, element), line)
                        slots.append(slot)
                    self.tuples[name] = slots
                    continue
                if isinstance(node.value, ast.Name) and node.value.id in self.tuples:
                    name = node.targets[0].id
                    if self.control_depth:
                        raise PythonSubsetError(f"line {line}: tuple copying inside a branch or loop is unsupported")
                    slots = []
                    for element in self.tuples[node.value.id]:
                        slot = self.temp()
                        self.emit(Assign(slot, element), line)
                        slots.append(slot)
                    self.tuples[name] = slots
                    continue
                if self.control_depth and node.targets[0].id in self.tuples:
                    raise PythonSubsetError(f"line {line}: replacing a tuple with a scalar inside a branch or loop is unsupported")
                self.tuples.pop(node.targets[0].id, None)
                value = node.value
                if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id != "print":
                    if value.keywords:
                        raise PythonSubsetError(f"line {line}: keyword arguments are unsupported")
                    if value.func.id in PARALLEL_REDUCTIONS:
                        self.emit(Assign(node.targets[0].id, self.expr(value)), line)
                    else:
                        self.emit(Call(node.targets[0].id, value.func.id, [self.expr(a) for a in value.args]), line)
                else:
                    self.emit(Assign(node.targets[0].id, self.expr(value)), line)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
                self.emit(Assign(node.target.id, self.expr(node.value)), line)
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and type(node.op) in self.OPS:
                self.emit(BinOp(node.target.id, self.OPS[type(node.op)], node.target.id, self.expr(node.value)), line)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if call.keywords:
                    raise PythonSubsetError(f"line {line}: keyword arguments are unsupported")
                if isinstance(call.func, ast.Name) and call.func.id == "print":
                    if len(call.args) != 1:
                        raise PythonSubsetError(f"line {line}: print supports exactly one value")
                    for arg in call.args:
                        self.emit(Print(self.expr(arg)), line)
                elif isinstance(call.func, ast.Name):
                    if call.func.id in PARALLEL_REDUCTIONS:
                        self.expr(call)
                    else:
                        self.emit(Call(None, call.func.id, [self.expr(a) for a in call.args]), line)
                else:
                    raise PythonSubsetError(f"line {line}: only direct function calls are supported")
            elif isinstance(node, ast.Return):
                self.emit(Return(self.expr(node.value) if node.value else None), line)
            elif isinstance(node, ast.If):
                otherwise, done = self.label("else"), self.label("endif")
                self.emit(IfZ(self.expr(node.test), otherwise if node.orelse else done), line)
                self.control_depth += 1
                self.block(node.body)
                self.control_depth -= 1
                if node.orelse:
                    self.emit(Goto(done), line)
                    self.emit(Label(otherwise), line)
                    self.control_depth += 1
                    self.block(node.orelse)
                    self.control_depth -= 1
                self.emit(Label(done), line)
            elif isinstance(node, ast.While):
                if node.orelse:
                    raise PythonSubsetError(f"line {line}: while-else is unsupported")
                top, done = self.label("while"), self.label("endwhile")
                self.emit(Label(top), line)
                self.emit(IfZ(self.expr(node.test), done), line)
                self.control_depth += 1
                self.block(node.body)
                self.control_depth -= 1
                self.emit(Goto(top), line)
                self.emit(Label(done), line)
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                if node.orelse:
                    raise PythonSubsetError(f"line {line}: for-else is unsupported")
                if node.iter.keywords or not 1 <= len(node.iter.args) <= 3:
                    raise PythonSubsetError(f"line {line}: range supports 1 to 3 positional arguments")
                values = [self.expr(a) for a in node.iter.args[:2]]
                if len(values) == 1:
                    start, stop, step = 0, values[0], 1
                else:
                    start, stop = values
                    step = 1
                    if len(node.iter.args) == 3:
                        step_node = node.iter.args[2]
                        if isinstance(step_node, ast.Constant) and type(step_node.value) is int:
                            step = step_node.value
                        elif isinstance(step_node, ast.UnaryOp) and isinstance(step_node.op, ast.USub) and isinstance(step_node.operand, ast.Constant) and type(step_node.operand.value) is int:
                            step = -step_node.operand.value
                        else:
                            raise PythonSubsetError(f"line {line}: range step must be an integer literal")
                if step == 0:
                    raise PythonSubsetError(f"line {line}: range step must be a nonzero integer literal")
                self.emit(Assign(node.target.id, start), line)
                end = self.temp()
                self.emit(Assign(end, stop), line)
                top, done = self.label("for"), self.label("endfor")
                test = self.temp()
                self.emit(Label(top), line)
                self.emit(BinOp(test, "<" if step > 0 else ">", node.target.id, end), line)
                self.emit(IfZ(test, done), line)
                self.control_depth += 1
                self.block(node.body)
                self.control_depth -= 1
                self.emit(BinOp(node.target.id, "+", node.target.id, step), line)
                self.emit(Goto(top), line)
                self.emit(Label(done), line)
            elif isinstance(node, ast.Pass):
                continue
            else:
                raise PythonSubsetError(f"line {line}: unsupported statement: {ast.unparse(node)}")

    def lower(self, body):
        self.block(body)
        return self.fn


def _is_main_guard(node):
    return (isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == "__main__"
            and not node.orelse)


def parse_python(source):
    """Parse supported Python source and return IR functions, including an entry main."""
    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        raise PythonSubsetError(f"line {exc.lineno}: {exc.msg}") from None
    functions, top_level, function_names = [], [], set()
    reserved = {node.id for node in ast.walk(module) if isinstance(node, ast.Name)}
    for stmt in module.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue
        if isinstance(stmt, ast.FunctionDef):
            if stmt.name in function_names:
                raise PythonSubsetError(f"line {stmt.lineno}: function '{stmt.name}' is defined more than once")
            function_names.add(stmt.name)
            if stmt.decorator_list or stmt.args.vararg or stmt.args.kwarg or stmt.args.kwonlyargs or stmt.args.defaults:
                raise PythonSubsetError(f"line {stmt.lineno}: decorators and advanced function arguments are unsupported")
            if stmt.name != "main" and not _always_returns(stmt.body):
                raise PythonSubsetError(f"line {stmt.lineno}: function '{stmt.name}' must return an integer on every path")
            if stmt.name != "main" and any(isinstance(n, ast.Return) and n.value is None for n in ast.walk(stmt)):
                raise PythonSubsetError(f"line {stmt.lineno}: function '{stmt.name}' cannot use a bare return")
            params = [arg.arg for arg in stmt.args.args]
            _validate_assignments(stmt.body, set(params))
            functions.append(_Lowerer(stmt.name, params, reserved).lower(stmt.body))
        elif _is_main_guard(stmt):
            # Accept the common Python script entry-point guard.
            if not (len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Expr)
                    and isinstance(stmt.body[0].value, ast.Call)
                    and isinstance(stmt.body[0].value.func, ast.Name)
                    and stmt.body[0].value.func.id == "main"):
                top_level.extend(stmt.body)
        else:
            top_level.append(stmt)
    if any(fn.name == "main" for fn in functions):
        if top_level:
            raise PythonSubsetError("top-level executable statements cannot be mixed with a defined main() function")
    else:
        _validate_assignments(top_level, set())
        functions.append(_Lowerer("main", [], reserved).lower(top_level))
    return functions
