"""Instruction set, binary bytecode format and disassembler for the CVM stack machine.

Binary file layout (little-endian):
    magic  "CVM1"            4 bytes
    count  uint32            number of instructions
    then `count` fixed-width instructions of 9 bytes each:
        opcode  uint8
        a       int32   (first operand, 0 if unused)
        b       int32   (second operand, 0 if unused)
Jump / call targets are instruction indices.
"""
import struct
from enum import IntEnum
from typing import NamedTuple


class Op(IntEnum):
    # stack
    PUSH = 1    # PUSH n        push constant n
    POP = 2     # POP           discard top
    DUP = 3     # DUP           duplicate top
    # variables (slots in the current frame)
    LOAD = 4    # LOAD s        push locals[s]
    STORE = 5   # STORE s       locals[s] = pop
    # arithmetic (pop b, pop a, push a OP b)
    ADD = 6
    SUB = 7
    MUL = 8
    DIV = 9     # truncating division, traps on zero
    MOD = 10    # traps on zero
    # comparisons (push 1 or 0)
    LT = 11
    LE = 12
    GT = 13
    GE = 14
    EQ = 15
    NE = 16
    # control flow
    JMP = 17    # JMP addr      unconditional jump
    JZ = 18     # JZ addr       pop; jump if == 0
    JNZ = 19    # JNZ addr      pop; jump if != 0
    CALL = 20   # CALL addr, n  pop n args into a new frame, jump to addr
    ENTER = 21  # ENTER n       size the current frame to n local slots
    RET = 22    # RET           pop return value, drop frame, push value for caller
    # misc
    PRINT = 23  # PRINT         pop and print
    HALT = 24   # HALT          stop the machine


class Instr(NamedTuple):
    op: Op
    a: int = 0
    b: int = 0


OPERANDS = {
    Op.PUSH: 1, Op.LOAD: 1, Op.STORE: 1, Op.JMP: 1, Op.JZ: 1,
    Op.JNZ: 1, Op.ENTER: 1, Op.CALL: 2,
}
JUMPS = {Op.JMP, Op.JZ, Op.JNZ}

MAGIC = b"CVM1"
_INSTR = struct.Struct("<Bii")


def encode(code):
    """Instruction list -> bytes."""
    out = bytearray(MAGIC)
    out += struct.pack("<I", len(code))
    for ins in code:
        out += _INSTR.pack(int(ins.op), ins.a, ins.b)
    return bytes(out)


def decode(data):
    """bytes -> instruction list. Raises ValueError on a malformed file."""
    if data[:4] != MAGIC:
        raise ValueError("not a CVM bytecode file (bad magic number)")
    (count,) = struct.unpack_from("<I", data, 4)
    if len(data) != 8 + count * _INSTR.size:
        raise ValueError("corrupt bytecode file (size mismatch)")
    code = []
    for i in range(count):
        opv, a, b = _INSTR.unpack_from(data, 8 + i * _INSTR.size)
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
    if n == 2:
        return f"{ins.op.name:<6} {ins.a}, {ins.b}"
    return ins.op.name


def disassemble(code, funcs=None, labels=None):
    """Readable listing. `funcs`: addr->name, `labels`: addr->[names] (both optional)."""
    funcs, labels = funcs or {}, labels or {}
    lines = []
    for addr, ins in enumerate(code):
        if addr in funcs:
            lines.append(f"{funcs[addr]}:")
        for name in labels.get(addr, []):
            lines.append(f"  {name}:")
        text = format_instr(ins)
        if ins.op == Op.CALL and ins.a in funcs:
            text += f"    ; -> {funcs[ins.a]}"
        elif ins.op in JUMPS and ins.a in labels:
            text += f"    ; -> {labels[ins.a][0]}"
        lines.append(f"    {addr:04d}  {text}")
    return "\n".join(lines)
