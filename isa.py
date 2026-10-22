import struct
from enum import IntEnum
from typing import NamedTuple


class Op(IntEnum):
    PUSH = 1
    POP = 2
    DUP = 3
    LOAD = 4
    STORE = 5
    ADD = 6
    SUB = 7
    MUL = 8
    DIV = 9
    MOD = 10
    LT = 11
    LE = 12
    GT = 13
    GE = 14
    EQ = 15
    NE = 16
    JMP = 17
    JZ = 18
    JNZ = 19
    CALL = 20
    ENTER = 21
    RET = 22
    PRINT = 23
    HALT = 24
    PAR_SUM = 25
    FLOOR_DIV = 26
    PY_MOD = 27
    PAR_REDUCE = 28


class Instr(NamedTuple):
    op: Op
    a: int = 0
    b: int = 0


OPERANDS = {
    Op.PUSH: 1, Op.LOAD: 1, Op.STORE: 1, Op.JMP: 1, Op.JZ: 1,
    Op.JNZ: 1, Op.ENTER: 1, Op.CALL: 2, Op.PAR_SUM: 2, Op.PAR_REDUCE: 2,
}

JUMPS = {Op.JMP, Op.JZ, Op.JNZ}

REDUCTION_MODES = {
    "min": 1, "max": 2, "product": 3, "count": 4,
    "any": 5, "all": 6, "bit_and": 7, "bit_or": 8,
    "bit_xor": 9, "gcd": 10, "lcm": 11, "average": 12,
    "argmin": 13, "argmax": 14,
}
REDUCTION_NAMES = {mode: name for name, mode in REDUCTION_MODES.items()}

MAGIC = b"CVM"
VERSION = 1
LEGACY_MAGIC = b"CVM1"
_INSTR = struct.Struct("<Bii")


def encode(code):
    out = bytearray(MAGIC + bytes((VERSION,)))
    out += struct.pack("<I", len(code))
    for ins in code:
        out += _INSTR.pack(int(ins.op), ins.a, ins.b)
    return bytes(out)


def decode(data):
    if data[:4] == LEGACY_MAGIC:
        # Compatibility with files produced by the pre-versioned CVM1 writer.
        header_size = 8
    elif data[:3] == MAGIC:
        if len(data) < 8:
            raise ValueError("corrupt bytecode file (truncated header)")
        version = data[3]
        if version != VERSION:
            raise ValueError(f"unsupported bytecode version {version}; this VM supports version {VERSION}")
        header_size = 8
    else:
        raise ValueError("not a CVM bytecode file (bad magic number)")
    (count,) = struct.unpack_from("<I", data, 4)
    if len(data) != header_size + count * _INSTR.size:
        raise ValueError("corrupt bytecode file (size mismatch)")
    code = []
    for i in range(count):
        opv, a, b = _INSTR.unpack_from(data, header_size + i * _INSTR.size)
        try:
            op = Op(opv)
        except ValueError:
            raise ValueError(f"invalid opcode {opv} at instruction {i}") from None
        code.append(Instr(op, a, b))
    return code


def format_instr(ins):
    n = OPERANDS.get(ins.op, 0)
    if n == 1:
        return f"{ins.op.name:<6} {ins.a}"
    if ins.op == Op.PAR_REDUCE:
        return f"{ins.op.name:<6} {ins.a}, {REDUCTION_NAMES.get(ins.b, ins.b)}"
    if n == 2:
        return f"{ins.op.name:<6} {ins.a}, {ins.b}"
    return ins.op.name


def disassemble(code, funcs=None, labels=None):
    funcs, labels = funcs or {}, labels or {}
    lines = []
    for addr, ins in enumerate(code):
        if addr in funcs:
            lines.append(f"{funcs[addr]}:")
        for name in labels.get(addr, []):
            lines.append(f"  {name}:")
        text = format_instr(ins)
        if ins.op in (Op.CALL, Op.PAR_SUM, Op.PAR_REDUCE) and ins.a in funcs:
            text += f"    ; -> {funcs[ins.a]}"
        elif ins.op in JUMPS and ins.a in labels:
            text += f"    ; -> {labels[ins.a][0]}"
        lines.append(f"    {addr:04d}  {text}")
    return "\n".join(lines)
