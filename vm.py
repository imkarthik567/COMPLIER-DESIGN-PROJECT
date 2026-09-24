"""CVM: a stack-based virtual machine.

State
    code    list of Instr (read-only)
    ip      index of the next instruction
    stack   operand stack shared by all frames
    frames  call stack; each Frame has locals, return address and the stack
            height at entry (`base`) so a callee can never touch its caller's
            operands
Traps (VMError): stack underflow, division by zero, bad slot, bad jump,
call-stack overflow, step limit (catches infinite loops).
"""
from dataclasses import dataclass

from isa import Op, format_instr


class VMError(Exception):
    pass


@dataclass
class Frame:
    ret_ip: int
    locals: list
    base: int


def _div(a, b):                      # truncating, like C
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


_BINARY = {
    Op.ADD: lambda a, b: a + b,
    Op.SUB: lambda a, b: a - b,
    Op.MUL: lambda a, b: a * b,
    Op.LT: lambda a, b: int(a < b),
    Op.LE: lambda a, b: int(a <= b),
    Op.GT: lambda a, b: int(a > b),
    Op.GE: lambda a, b: int(a >= b),
    Op.EQ: lambda a, b: int(a == b),
    Op.NE: lambda a, b: int(a != b),
}


class VM:
    def __init__(self, code, trace=False, max_steps=5_000_000, max_depth=1000, out=print):
        self.code = code
        self.trace = trace
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.out = out
        self.reset()

    def reset(self):
        self.stack, self.frames = [], []
        self.ip = self.pc = self.steps = 0
        self.output = []

    # ------------------------------------------------------------ helpers
    def _err(self, msg):
        op = self.code[self.pc].op.name if 0 <= self.pc < len(self.code) else "?"
        return VMError(f"{msg}  [pc={self.pc}, op={op}, call depth={len(self.frames)}]")

    def _pop(self):
        base = self.frames[-1].base if self.frames else 0
        if len(self.stack) <= base:
            raise self._err("stack underflow")
        return self.stack.pop()

    def _locals(self):
        if not self.frames:
            raise self._err("no active frame")
        return self.frames[-1].locals

    def _trace(self, ins):
        shown = self.stack[-6:]
        more = "... " if len(self.stack) > 6 else ""
        print(f"  {self.pc:04d}  {format_instr(ins):<14} d={len(self.frames)}  stack=[{more}{', '.join(map(str, shown))}]")

    # ---------------------------------------------------------------- run
    def run(self):
        """Execute until HALT. Returns the list of printed values."""
        self.reset()
        code = self.code
        while True:
            if not 0 <= self.ip < len(code):
                raise VMError(f"instruction pointer out of range: {self.ip}")
            self.pc = self.ip
            ins = code[self.pc]
            op = ins.op
            self.steps += 1
            if self.steps > self.max_steps:
                raise self._err("step limit exceeded (possible infinite loop)")
            if self.trace:
                self._trace(ins)
            self.ip += 1

            if op == Op.PUSH:
                self.stack.append(ins.a)
            elif op == Op.POP:
                self._pop()
            elif op == Op.DUP:
                v = self._pop()
                self.stack += [v, v]
            elif op == Op.LOAD:
                loc = self._locals()
                if not 0 <= ins.a < len(loc):
                    raise self._err(f"bad local slot {ins.a}")
                self.stack.append(loc[ins.a])
            elif op == Op.STORE:
                loc = self._locals()
                v = self._pop()
                if not 0 <= ins.a < len(loc):
                    raise self._err(f"bad local slot {ins.a}")
                loc[ins.a] = v
            elif op in _BINARY:
                b = self._pop()
                a = self._pop()
                self.stack.append(_BINARY[op](a, b))
            elif op in (Op.DIV, Op.MOD):
                b = self._pop()
                a = self._pop()
                if b == 0:
                    raise self._err("division by zero")
                q = _div(a, b)
                self.stack.append(q if op == Op.DIV else a - b * q)
            elif op == Op.JMP:
                self.ip = ins.a
            elif op == Op.JZ:
                if self._pop() == 0:
                    self.ip = ins.a
            elif op == Op.JNZ:
                if self._pop() != 0:
                    self.ip = ins.a
            elif op == Op.CALL:
                if len(self.frames) >= self.max_depth:
                    raise self._err("call stack overflow")
                base = self.frames[-1].base if self.frames else 0
                n = ins.b
                if len(self.stack) - base < n:
                    raise self._err("stack underflow while passing arguments")
                args = self.stack[len(self.stack) - n:]
                del self.stack[len(self.stack) - n:]
                self.frames.append(Frame(self.ip, args, len(self.stack)))
                self.ip = ins.a
            elif op == Op.ENTER:
                loc = self._locals()
                loc.extend([0] * (ins.a - len(loc)))
            elif op == Op.RET:
                if not self.frames:
                    raise self._err("RET outside of a function")
                v = self._pop()
                fr = self.frames.pop()
                del self.stack[fr.base:]
                self.stack.append(v)
                self.ip = fr.ret_ip
            elif op == Op.PRINT:
                v = self._pop()
                self.output.append(v)
                self.out(v)
            elif op == Op.HALT:
                return self.output
            else:
                raise self._err("illegal instruction")
