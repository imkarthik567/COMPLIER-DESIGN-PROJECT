"""Three-address intermediate representation (the input of the backend).

Text syntax (one statement per line, '#' starts a comment):

    func name(p1, p2):        start of a function
    x = 5                     copy (constant or variable)
    x = a + b                 binary op: + - * / % < <= > >= == !=
    x = call f(a, b)          call, result stored in x
    call f(a)                 call, result discarded
    label:                    jump target
    goto label
    ifz  c goto label         jump if c == 0
    ifnz c goto label         jump if c != 0
    print a
    return [a]
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional, Union

Operand = Union[int, str]


class IRError(Exception):
    """Raised when an error occurs during IR parsing or validation."""
    pass


class _Node:
    """Base class for all intermediate representation (IR) syntax nodes."""
    line = 0  # source line, filled in by the parser


@dataclass
class Assign(_Node):
    dst: str
    src: Operand


@dataclass
class BinOp(_Node):
    dst: str
    op: str
    a: Operand
    b: Operand


@dataclass
class Label(_Node):
    name: str


@dataclass
class Goto(_Node):
    target: str


@dataclass
class IfZ(_Node):
    cond: Operand
    target: str


@dataclass
class IfNz(_Node):
    cond: Operand
    target: str


@dataclass
class Print(_Node):
    arg: Operand


@dataclass
class Return(_Node):
    arg: Optional[Operand] = None


@dataclass
class Call(_Node):
    dst: Optional[str]
    func: str
    args: list = field(default_factory=list)


@dataclass
class LoadIndexed(_Node):
    dst: str
    array: str
    index: Operand


@dataclass
class StoreIndexed(_Node):
    array: str
    index: Operand
    src: Operand


@dataclass
class NewArray(_Node):
    dst: str
    size: Operand


@dataclass
class VLoad(_Node):
    dst: str
    array: str
    base_index: Operand
    width: int = 4


@dataclass
class VStore(_Node):
    array: str
    base_index: Operand
    src: Operand
    width: int = 4


@dataclass
class VAdd(_Node):
    dst: str
    a: Operand
    b: Operand


@dataclass
class VSub(_Node):
    dst: str
    a: Operand
    b: Operand


@dataclass
class VMul(_Node):
    dst: str
    a: Operand
    b: Operand


@dataclass
class VDiv(_Node):
    dst: str
    a: Operand
    b: Operand


@dataclass
class Function:
    name: str
    params: List[str]
    body: list
    line: int = 0


# ---------------------------------------------------------------- parsing
IDENT = r"[A-Za-z_]\w*"
OPND = rf"(?:-?\d+|{IDENT})"
IDENT_RE = re.compile(IDENT)
NUM_RE = re.compile(r"-?\d+")

FUNC_RE = re.compile(rf"^func\s+({IDENT})\s*\(([^)]*)\)\s*:$")
LABEL_RE = re.compile(rf"^({IDENT})\s*:$")
GOTO_RE = re.compile(rf"^goto\s+({IDENT})$")
IF_RE = re.compile(rf"^(ifz|ifnz|if)\s+({OPND})\s+goto\s+({IDENT})$")
PRINT_RE = re.compile(rf"^print\s+({OPND})$")
RETURN_RE = re.compile(rf"^return(?:\s+({OPND}))?$")
CALL_RE = re.compile(rf"^(?:({IDENT})\s*=\s*)?call\s+({IDENT})\s*\(([^)]*)\)$")
BINOP_RE = re.compile(rf"^({IDENT})\s*=\s*({OPND})\s*(<=|>=|==|!=|[-+*/%<>])\s*({OPND})$")
ASSIGN_RE = re.compile(rf"^({IDENT})\s*=\s*({OPND})$")
LOAD_INDEXED_RE = re.compile(rf"^({IDENT})\s*=\s*({IDENT})\s*\[\s*({OPND})\s*\]$")
STORE_INDEXED_RE = re.compile(rf"^({IDENT})\s*\[\s*({OPND})\s*\]\s*=\s*({OPND})$")
NEW_ARRAY_RE = re.compile(rf"^({IDENT})\s*=\s*(?:new_array|array)\s*\(\s*({OPND})\s*\)$")
VLOAD_RE = re.compile(rf"^({IDENT})\s*=\s*vload\s*\(?\s*({IDENT})\s*,\s*({OPND})\s*,\s*(\d+)\s*\)?$")
VSTORE_RE = re.compile(rf"^vstore\s*\(?\s*({IDENT})\s*,\s*({OPND})\s*,\s*({OPND})\s*,\s*(\d+)\s*\)?$")
VBINOP_RE = re.compile(rf"^({IDENT})\s*=\s*(vadd|vsub|vmul|vdiv)\s*\(?\s*({OPND})\s*,\s*({OPND})\s*\)?$")


def _fail(n, msg):
    raise IRError(f"line {n}: {msg}")


def _operand(tok, n):
    tok = tok.strip()
    if NUM_RE.fullmatch(tok):
        return int(tok)
    if IDENT_RE.fullmatch(tok):
        return tok
    _fail(n, f"bad operand {tok!r}")


def _parse_instr(line, n):
    m = LABEL_RE.match(line)
    if m:
        return Label(m.group(1))
    m = GOTO_RE.match(line)
    if m:
        return Goto(m.group(1))
    m = IF_RE.match(line)
    if m:
        cls = IfZ if m.group(1) == "ifz" else IfNz
        return cls(_operand(m.group(2), n), m.group(3))
    m = PRINT_RE.match(line)
    if m:
        return Print(_operand(m.group(1), n))
    m = RETURN_RE.match(line)
    if m:
        return Return(_operand(m.group(1), n) if m.group(1) else None)
    m = CALL_RE.match(line)
    if m:
        args = [_operand(a, n) for a in m.group(3).split(",") if a.strip()]
        return Call(m.group(1), m.group(2), args)
    m = BINOP_RE.match(line)
    if m:
        return BinOp(m.group(1), m.group(3), _operand(m.group(2), n), _operand(m.group(4), n))
    m = LOAD_INDEXED_RE.match(line)
    if m:
        return LoadIndexed(m.group(1), m.group(2), _operand(m.group(3), n))
    m = STORE_INDEXED_RE.match(line)
    if m:
        return StoreIndexed(m.group(1), _operand(m.group(2), n), _operand(m.group(3), n))
    m = NEW_ARRAY_RE.match(line)
    if m:
        return NewArray(m.group(1), _operand(m.group(2), n))
    m = VLOAD_RE.match(line)
    if m:
        return VLoad(m.group(1), m.group(2), _operand(m.group(3), n), int(m.group(4)))
    m = VSTORE_RE.match(line)
    if m:
        return VStore(m.group(1), _operand(m.group(2), n), _operand(m.group(3), n), int(m.group(4)))
    m = VBINOP_RE.match(line)
    if m:
        op = m.group(2).lower()
        v_cls = {"vadd": VAdd, "vsub": VSub, "vmul": VMul, "vdiv": VDiv}[op]
        return v_cls(m.group(1), _operand(m.group(3), n), _operand(m.group(4), n))
    m = ASSIGN_RE.match(line)
    if m:
        return Assign(m.group(1), _operand(m.group(2), n))
    _fail(n, f"cannot parse {line!r}")


def parse(text):
    """IR text -> list[Function]."""
    funcs, cur, seen = [], None, set()
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = FUNC_RE.match(line)
        if m:
            name = m.group(1)
            params = [p.strip() for p in m.group(2).split(",") if p.strip()]
            for p in params:
                if not IDENT_RE.fullmatch(p):
                    _fail(n, f"bad parameter name {p!r}")
            if len(set(params)) != len(params):
                _fail(n, "duplicate parameter name")
            if name in seen:
                _fail(n, f"function '{name}' defined twice")
            seen.add(name)
            cur = Function(name, params, [], n)
            funcs.append(cur)
            continue
        if cur is None:
            _fail(n, "instruction outside of a function")
        ins = _parse_instr(line, n)
        ins.line = n
        cur.body.append(ins)
    if not funcs:
        raise IRError("empty program")
    return funcs


# ---------------------------------------------------------------- printing
def fmt(ins):
    """Format an individual IR instruction into readable three-address code syntax."""
    if isinstance(ins, Assign):
        return f"{ins.dst} = {ins.src}"
    if isinstance(ins, BinOp):
        return f"{ins.dst} = {ins.a} {ins.op} {ins.b}"
    if isinstance(ins, Label):
        return f"{ins.name}:"
    if isinstance(ins, Goto):
        return f"goto {ins.target}"
    if isinstance(ins, IfZ):
        return f"ifz {ins.cond} goto {ins.target}"
    if isinstance(ins, IfNz):
        return f"ifnz {ins.cond} goto {ins.target}"
    if isinstance(ins, Print):
        return f"print {ins.arg}"
    if isinstance(ins, Return):
        return "return" if ins.arg is None else f"return {ins.arg}"
    if isinstance(ins, Call):
        call = f"call {ins.func}({', '.join(map(str, ins.args))})"
        return f"{ins.dst} = {call}" if ins.dst else call
    if isinstance(ins, LoadIndexed):
        return f"{ins.dst} = {ins.array}[{ins.index}]"
    if isinstance(ins, StoreIndexed):
        return f"{ins.array}[{ins.index}] = {ins.src}"
    if isinstance(ins, NewArray):
        return f"{ins.dst} = new_array({ins.size})"
    if isinstance(ins, VLoad):
        return f"{ins.dst} = vload {ins.array}, {ins.base_index}, {ins.width}"
    if isinstance(ins, VStore):
        return f"vstore {ins.array}, {ins.base_index}, {ins.src}, {ins.width}"
    if isinstance(ins, VAdd):
        return f"{ins.dst} = vadd {ins.a}, {ins.b}"
    if isinstance(ins, VSub):
        return f"{ins.dst} = vsub {ins.a}, {ins.b}"
    if isinstance(ins, VMul):
        return f"{ins.dst} = vmul {ins.a}, {ins.b}"
    if isinstance(ins, VDiv):
        return f"{ins.dst} = vdiv {ins.a}, {ins.b}"
    raise TypeError(ins)


def format_ir(functions):
    """Format a list of IR functions into three-address code string."""
    lines = []
    for f in functions:
        lines.append(f"func {f.name}({', '.join(f.params)}):")
        for ins in f.body:
            pad = "  " if isinstance(ins, Label) else "    "
            lines.append(pad + fmt(ins))
    return "\n".join(lines)


# ------------------------------------------------- basic blocks and CFG
@dataclass
class Block:
    id: int
    instrs: list
    succs: list = field(default_factory=list)
    preds: list = field(default_factory=list)


def basic_blocks(fn):
    """Split a function into basic blocks and compute successor edges."""
    body = fn.body
    if not body:
        return []
    leaders = {0}
    for i, ins in enumerate(body):
        if isinstance(ins, Label):
            leaders.add(i)
        if isinstance(ins, (Goto, IfZ, IfNz, Return)) and i + 1 < len(body):
            leaders.add(i + 1)
    starts = sorted(leaders)
    blocks = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(body)
        blocks.append(Block(k, body[s:e]))

    label_block = {}
    for b in blocks:
        for ins in b.instrs:
            if isinstance(ins, Label):
                label_block[ins.name] = b.id
            else:
                break

    def target(name, line):
        if name not in label_block:
            raise IRError(f"line {line}: undefined label '{name}' in function '{fn.name}'")
        return label_block[name]

    for b in blocks:
        last = b.instrs[-1]
        nxt = [b.id + 1] if b.id + 1 < len(blocks) else []
        if isinstance(last, Goto):
            b.succs = [target(last.target, last.line)]
        elif isinstance(last, (IfZ, IfNz)):
            b.succs = [target(last.target, last.line)] + nxt
        elif isinstance(last, Return):
            b.succs = []
        else:
            b.succs = nxt

    for b in blocks:
        for s in b.succs:
            if s < len(blocks) and b.id not in blocks[s].preds:
                blocks[s].preds.append(b.id)

    return blocks


def build_cfg(fn):
    """Convenience alias: construct { block_label: BasicBlock } CFG dict."""
    from optimizer import build_cfg as _build_cfg
    return _build_cfg(fn)


def format_cfg(functions):
    """Format basic blocks and control flow graph edges into readable text."""
    lines = []
    for fn in functions:
        blocks = basic_blocks(fn)
        lines.append(f"function {fn.name}: {len(blocks)} basic block(s)")
        for b in blocks:
            succ = ", ".join(f"B{s}" for s in b.succs) or "(exit)"
            lines.append(f"  B{b.id}  -> {succ}")
            for ins in b.instrs:
                lines.append(f"        {fmt(ins)}")
    return "\n".join(lines)
