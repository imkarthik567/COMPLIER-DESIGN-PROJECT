"""Small, conservative IR optimizations used by the CVM compiler."""
from concurrent.futures import ThreadPoolExecutor

from ir import Assign, BinOp, Call, Goto, IfNz, IfZ, Label, ParallelSum, ParallelReduce, Print, Return, Function, basic_blocks

_TOP = object()
_UNKNOWN = object()


def _value(x, env):
    seen = set()
    while isinstance(x, str) and x in env and x not in seen:
        seen.add(x)
        x = env[x]
    return x


def _invalidate_aliases(env, name):
    """Drop cached copies whose source would change when name is redefined."""
    stale = []
    for alias in env:
        current, seen = alias, set()
        while isinstance(current, str) and current in env and current not in seen:
            seen.add(current)
            current = env[current]
            if current == name:
                stale.append(alias)
                break
    for alias in stale:
        env.pop(alias, None)


def _keep_line(replacement, original):
    replacement.line = getattr(original, "line", 0)
    return replacement


def _bytecode_constant_fits(value):
    return -(2**31) <= value < 2**31


def _fold(op, a, b):
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/": return abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1)
    if op == "%": return a - (abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1)) * b
    if op == "//": return a // b
    if op == "%%": return a % b
    if op == "<": return int(a < b)
    if op == "<=": return int(a <= b)
    if op == ">": return int(a > b)
    if op == ">=": return int(a >= b)
    if op == "==": return int(a == b)
    if op == "!=": return int(a != b)
    raise ValueError(op)


def _transfer_facts(instructions, incoming):
    """Apply a block in the constant-propagation lattice (TOP/unknown/constant)."""
    facts = dict(incoming)
    for ins in instructions:
        if isinstance(ins, Assign):
            value = facts.get(ins.src, _UNKNOWN) if isinstance(ins.src, str) else ins.src
            if isinstance(value, int) and _bytecode_constant_fits(value):
                facts[ins.dst] = value
            else:
                facts[ins.dst] = value if value is _TOP else _UNKNOWN
        elif isinstance(ins, BinOp):
            a = facts.get(ins.a, _UNKNOWN) if isinstance(ins.a, str) else ins.a
            b = facts.get(ins.b, _UNKNOWN) if isinstance(ins.b, str) else ins.b
            if (isinstance(a, int) and isinstance(b, int)
                    and not (ins.op in ("/", "%", "//", "%%") and b == 0)):
                result = _fold(ins.op, a, b)
                if _bytecode_constant_fits(result):
                    facts[ins.dst] = result
                else:
                    facts[ins.dst] = _UNKNOWN
            else:
                facts[ins.dst] = _UNKNOWN
        elif isinstance(ins, (Call, ParallelSum, ParallelReduce)) and ins.dst:
            facts[ins.dst] = _UNKNOWN
    return facts


def _cfg_constant_propagation(fn):
    """Solve a forward must-constant analysis over the function's CFG.

    A fact survives a join only when every incoming predecessor agrees. The
    analysis starts non-entry blocks at the lattice's optimistic TOP value and
    iterates to a fixed point; conflicting or unknown loop back-edge facts
    remove constants before the transformed IR is emitted.
    """
    blocks = basic_blocks(fn)
    if not blocks:
        return fn
    predecessors = {b.id: [] for b in blocks}
    for block in blocks:
        for successor in block.succs:
            predecessors[successor].append(block.id)

    variables = set(fn.params)
    for block in blocks:
        for ins in block.instrs:
            if isinstance(ins, (Assign, BinOp)):
                variables.add(ins.dst)
            elif isinstance(ins, (Call, ParallelSum, ParallelReduce)) and ins.dst:
                variables.add(ins.dst)
    initial_top = {name: _TOP for name in variables}
    incoming = {b.id: dict(initial_top) for b in blocks}
    outgoing = {b.id: dict(initial_top) for b in blocks}
    changed = True
    while changed:
        changed = False
        for block in blocks:
            preds = predecessors[block.id]
            if block.id == 0 or not preds:
                merged = {name: _UNKNOWN for name in variables}
            else:
                first = outgoing[preds[0]]
                merged = {}
                for name in variables:
                    value = first.get(name, _UNKNOWN)
                    for pred in preds[1:]:
                        other = outgoing[pred].get(name, _UNKNOWN)
                        if value is _TOP:
                            value = other
                        elif other is not _TOP and other != value:
                            value = _UNKNOWN
                    merged[name] = value
            result = _transfer_facts(block.instrs, merged)
            if merged != incoming[block.id] or result != outgoing[block.id]:
                incoming[block.id], outgoing[block.id] = merged, result
                changed = True

    rewritten = []
    for block in blocks:
        facts = {name: value for name, value in incoming[block.id].items() if isinstance(value, int)}
        for ins in block.instrs:
            if isinstance(ins, Assign):
                src = _value(ins.src, facts)
                rewritten.append(_keep_line(Assign(ins.dst, src), ins))
                if isinstance(src, int) and _bytecode_constant_fits(src):
                    facts[ins.dst] = src
                else:
                    facts.pop(ins.dst, None)
            elif isinstance(ins, BinOp):
                a, b = _value(ins.a, facts), _value(ins.b, facts)
                rewritten.append(_keep_line(BinOp(ins.dst, ins.op, a, b), ins))
                facts = {name: value for name, value in
                         _transfer_facts([BinOp(ins.dst, ins.op, a, b)], facts).items()
                         if isinstance(value, int)}
            elif isinstance(ins, (IfZ, IfNz)):
                rewritten.append(_keep_line(type(ins)(_value(ins.cond, facts), ins.target), ins))
            elif isinstance(ins, Print):
                rewritten.append(_keep_line(Print(_value(ins.arg, facts)), ins))
            elif isinstance(ins, Return):
                rewritten.append(_keep_line(Return(_value(ins.arg, facts) if ins.arg is not None else None), ins))
            elif isinstance(ins, Call):
                rewritten.append(_keep_line(Call(ins.dst, ins.func, [_value(a, facts) for a in ins.args]), ins))
                if ins.dst:
                    facts.pop(ins.dst, None)
            elif isinstance(ins, ParallelSum):
                rewritten.append(_keep_line(ParallelSum(ins.dst, ins.func, _value(ins.start, facts), _value(ins.stop, facts)), ins))
                facts.pop(ins.dst, None)
            elif isinstance(ins, ParallelReduce):
                rewritten.append(_keep_line(ParallelReduce(ins.dst, ins.func, _value(ins.start, facts), _value(ins.stop, facts), ins.kind), ins))
                facts.pop(ins.dst, None)
            else:
                rewritten.append(ins)
    return Function(fn.name, list(fn.params), rewritten, fn.line)


def analyze_liveness(fn):
    """Return block live-in/live-out sets from a backward CFG fixed point."""
    blocks = basic_blocks(fn)
    uses, defs = {}, {}
    for block in blocks:
        use, defined = set(), set()
        for ins in block.instrs:
            if isinstance(ins, Assign):
                reads, writes = (ins.src,), (ins.dst,)
            elif isinstance(ins, BinOp):
                reads, writes = (ins.a, ins.b), (ins.dst,)
            elif isinstance(ins, (IfZ, IfNz)):
                reads, writes = (ins.cond,), ()
            elif isinstance(ins, Print):
                reads, writes = (ins.arg,), ()
            elif isinstance(ins, Return):
                reads, writes = (() if ins.arg is None else (ins.arg,)), ()
            elif isinstance(ins, Call):
                reads, writes = ins.args, (() if ins.dst is None else (ins.dst,))
            elif isinstance(ins, (ParallelSum, ParallelReduce)):
                reads, writes = (ins.start, ins.stop), (ins.dst,)
            else:
                reads, writes = (), ()
            use.update(x for x in reads if isinstance(x, str) and x not in defined)
            defined.update(writes)
        uses[block.id], defs[block.id] = use, defined

    live_in = {b.id: set() for b in blocks}
    live_out = {b.id: set() for b in blocks}
    changed = True
    while changed:
        changed = False
        for block in reversed(blocks):
            out = set().union(*(live_in[s] for s in block.succs)) if block.succs else set()
            incoming = uses[block.id] | (out - defs[block.id])
            if out != live_out[block.id] or incoming != live_in[block.id]:
                live_out[block.id], live_in[block.id] = out, incoming
                changed = True
    return {b.id: (live_in[b.id], live_out[b.id]) for b in blocks}


def _liveness_dead_code(fn):
    blocks = basic_blocks(fn)
    live = analyze_liveness(fn)
    rewritten = []
    for block in blocks:
        current = set(live[block.id][1])
        kept = []
        for ins in reversed(block.instrs):
            if isinstance(ins, Assign):
                if ins.dst not in current:
                    continue
                current.discard(ins.dst)
                if isinstance(ins.src, str): current.add(ins.src)
            elif isinstance(ins, BinOp):
                removable = ins.op not in ("/", "%", "//", "%%")
                if removable and ins.dst not in current:
                    continue
                current.discard(ins.dst)
                current.update(x for x in (ins.a, ins.b) if isinstance(x, str))
            elif isinstance(ins, (IfZ, IfNz)):
                if isinstance(ins.cond, str): current.add(ins.cond)
            elif isinstance(ins, Print):
                if isinstance(ins.arg, str): current.add(ins.arg)
            elif isinstance(ins, Return):
                if isinstance(ins.arg, str): current.add(ins.arg)
            elif isinstance(ins, Call):
                if ins.dst: current.discard(ins.dst)
                current.update(x for x in ins.args if isinstance(x, str))
            elif isinstance(ins, (ParallelSum, ParallelReduce)):
                current.discard(ins.dst)
                current.update(x for x in (ins.start, ins.stop) if isinstance(x, str))
            kept.append(ins)
        rewritten.extend(reversed(kept))
    return Function(fn.name, list(fn.params), rewritten, fn.line)


def _optimize_function(fn):
    """Run CFG constant analysis, then local simplification and dead-store cleanup."""
    fn = _cfg_constant_propagation(fn)
    env, out = {}, []
    for ins in fn.body:
        if isinstance(ins, Label):
            env.clear()
            out.append(ins)
        elif isinstance(ins, Assign):
            src = _value(ins.src, env)
            out.append(_keep_line(Assign(ins.dst, src), ins))
            _invalidate_aliases(env, ins.dst)
            env[ins.dst] = src
        elif isinstance(ins, BinOp):
            a, b = _value(ins.a, env), _value(ins.b, env)
            if isinstance(a, int) and isinstance(b, int) and not (ins.op in ("/", "%", "//", "%%") and b == 0):
                result = _fold(ins.op, a, b)
                if _bytecode_constant_fits(result):
                    out.append(_keep_line(Assign(ins.dst, result), ins))
                    _invalidate_aliases(env, ins.dst)
                    env[ins.dst] = result
                else:
                    # PUSH immediates are signed int32; keep the runtime operation
                    # so a wider computed Python integer does not become an invalid PUSH.
                    out.append(_keep_line(BinOp(ins.dst, ins.op, a, b), ins))
                    _invalidate_aliases(env, ins.dst)
                    env.pop(ins.dst, None)
                continue
            identity = None
            if ins.op == "+" and b == 0: identity = a
            elif ins.op == "+" and a == 0: identity = b
            elif ins.op == "-" and b == 0: identity = a
            elif ins.op == "*" and b == 1: identity = a
            elif ins.op == "*" and a == 1: identity = b
            elif ins.op == "*" and (a == 0 or b == 0): identity = 0
            elif ins.op == "/" and b == 1: identity = a
            if identity is not None:
                out.append(_keep_line(Assign(ins.dst, identity), ins))
                _invalidate_aliases(env, ins.dst)
                env[ins.dst] = identity
            else:
                out.append(_keep_line(BinOp(ins.dst, ins.op, a, b), ins))
                _invalidate_aliases(env, ins.dst)
                env.pop(ins.dst, None)
        elif isinstance(ins, (IfZ, IfNz)):
            out.append(_keep_line(type(ins)(_value(ins.cond, env), ins.target), ins))
            env.clear()
        elif isinstance(ins, Goto):
            out.append(ins)
            env.clear()
        elif isinstance(ins, Print):
            out.append(_keep_line(Print(_value(ins.arg, env)), ins))
        elif isinstance(ins, Return):
            out.append(_keep_line(Return(_value(ins.arg, env) if ins.arg is not None else None), ins))
            env.clear()
        elif isinstance(ins, Call):
            # Calls may have effects; discard facts for variables they could change.
            out.append(_keep_line(Call(ins.dst, ins.func, [_value(a, env) for a in ins.args]), ins))
            if ins.dst:
                _invalidate_aliases(env, ins.dst)
                env.pop(ins.dst, None)
        elif isinstance(ins, (ParallelSum, ParallelReduce)):
            start, stop = _value(ins.start, env), _value(ins.stop, env)
            reduced = (ParallelSum(ins.dst, ins.func, start, stop) if isinstance(ins, ParallelSum)
                       else ParallelReduce(ins.dst, ins.func, start, stop, ins.kind))
            out.append(_keep_line(reduced, ins))
            _invalidate_aliases(env, ins.dst)
            env.pop(ins.dst, None)
        else:
            out.append(ins)
    return _liveness_dead_code(Function(fn.name, list(fn.params), out, fn.line))


def optimize(functions, parallel=False):
    """Return optimized functions; parallel mode optimizes independent functions concurrently."""
    if parallel and len(functions) > 1:
        with ThreadPoolExecutor(max_workers=min(len(functions), 4)) as pool:
            return list(pool.map(_optimize_function, functions))
    return [_optimize_function(fn) for fn in functions]
