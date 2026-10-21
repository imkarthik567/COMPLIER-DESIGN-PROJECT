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
from dataclasses import dataclass, field

from isa import Op, format_instr


class VMError(Exception):
    """Raised when a runtime trap or virtual machine fault occurs."""
    pass


@dataclass
class Frame:
    ret_ip: int
    locals: list
    base: int
    regs: list = field(default_factory=lambda: [0] * 8)


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
    """Stack-based virtual machine with activation call frames and execution tracing."""

    def __init__(self, code, trace=False, max_steps=5_000_000, max_depth=1000, out=print):
        self.code = code
        self.trace = trace
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.out = out
        self.reset()

    def reset(self):
        """Reset operand stack, activation frames, registers, and instruction pointer."""
        self.stack, self.frames = [], []
        self.regs = [0] * 8
        self.ip = self.pc = self.steps = 0
        self.output = []

    # ------------------------------------------------------------ helpers
    def _err(self, msg):
        op = self.code[self.pc].op.name if 0 <= self.pc < len(self.code) else "?"
        return VMError(f"{msg}  [pc={self.pc}, op={op}, call depth={len(self.frames)}]")

    def _regs(self):
        return self.frames[-1].regs if self.frames else self.regs

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
                if 0 <= ins.a < 8:
                    self._regs()[ins.a] = v
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
                regs = self._regs()
                for i in range(min(len(loc), 8)):
                    regs[i] = loc[i]
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
            elif op == Op.RLOAD:
                rd, rs = ins.a, ins.b
                if not (0 <= rd < 8 and 0 <= rs < 8):
                    raise self._err(f"bad register in RLOAD: {rd}, {rs}")
                v = self._regs()[rs]
                self._regs()[rd] = v
                if self.frames and rd < len(self._locals()):
                    self._locals()[rd] = v
            elif op == Op.RSTORE:
                rd, mem = ins.a, ins.b
                if not 0 <= rd < 8:
                    raise self._err(f"bad register in RSTORE: {rd}")
                loc = self._locals()
                if not 0 <= mem < len(loc):
                    raise self._err(f"bad local slot {mem} in RSTORE")
                loc[mem] = self._regs()[rd]
            elif op == Op.RLOAD_IMM:
                rd, v = ins.a, ins.b
                if not 0 <= rd < 8:
                    raise self._err(f"bad register in RLOAD_IMM: {rd}")
                self._regs()[rd] = v
                if self.frames and rd < len(self._locals()):
                    self._locals()[rd] = v
            elif op == Op.LOAD_INDEXED:
                idx = self._pop()
                arr = self._locals()[ins.a]
                if not isinstance(arr, list):
                    raise self._err(f"expected array in slot {ins.a}")
                if idx < 0:
                    raise self._err(f"negative array index {idx}")
                if idx >= len(arr):
                    arr.extend([0] * (idx + 1 - len(arr)))
                self.stack.append(arr[idx])
            elif op == Op.STORE_INDEXED:
                v = self._pop()
                idx = self._pop()
                arr = self._locals()[ins.a]
                if not isinstance(arr, list):
                    raise self._err(f"expected array in slot {ins.a}")
                if idx < 0:
                    raise self._err(f"negative array index {idx}")
                if idx >= len(arr):
                    arr.extend([0] * (idx + 1 - len(arr)))
                arr[idx] = v
            elif op == Op.NEW_ARRAY:
                sz = self._pop()
                self.stack.append([0] * max(0, sz))
            elif op == Op.VLOAD:
                idx = self._pop()
                w = ins.b if ins.b > 0 else 4
                arr = self._locals()[ins.a]
                if not isinstance(arr, list):
                    raise self._err(f"expected array in slot {ins.a}")
                if idx < 0:
                    raise self._err(f"negative array index {idx}")
                if idx + w > len(arr):
                    arr.extend([0] * (idx + w - len(arr)))
                self.stack.append([arr[idx + k] for k in range(w)])
            elif op == Op.VSTORE:
                vec = self._pop()
                idx = self._pop()
                w = ins.b if ins.b > 0 else (len(vec) if isinstance(vec, list) else 4)
                arr = self._locals()[ins.a]
                if not isinstance(arr, list):
                    raise self._err(f"expected array in slot {ins.a}")
                if idx < 0:
                    raise self._err(f"negative array index {idx}")
                if idx + w > len(arr):
                    arr.extend([0] * (idx + w - len(arr)))
                if isinstance(vec, (int, float)):
                    vec = [vec] * w
                for k in range(w):
                    arr[idx + k] = vec[k]
            elif op == Op.VADD:
                b = self._pop()
                a = self._pop()
                if isinstance(a, (int, float)) and isinstance(b, list):
                    a = [a] * len(b)
                elif isinstance(b, (int, float)) and isinstance(a, list):
                    b = [b] * len(a)
                self.stack.append([x + y for x, y in zip(a, b)])
            elif op == Op.VSUB:
                b = self._pop()
                a = self._pop()
                if isinstance(a, (int, float)) and isinstance(b, list):
                    a = [a] * len(b)
                elif isinstance(b, (int, float)) and isinstance(a, list):
                    b = [b] * len(a)
                self.stack.append([x - y for x, y in zip(a, b)])
            elif op == Op.VMUL:
                b = self._pop()
                a = self._pop()
                if isinstance(a, (int, float)) and isinstance(b, list):
                    a = [a] * len(b)
                elif isinstance(b, (int, float)) and isinstance(a, list):
                    b = [b] * len(a)
                self.stack.append([x * y for x, y in zip(a, b)])
            elif op == Op.VDIV:
                b = self._pop()
                a = self._pop()
                if isinstance(a, (int, float)) and isinstance(b, list):
                    a = [a] * len(b)
                elif isinstance(b, (int, float)) and isinstance(a, list):
                    b = [b] * len(a)
                if any(y == 0 for y in b):
                    raise self._err("division by zero in vector operation")
                self.stack.append([_div(x, y) for x, y in zip(a, b)])
            else:
                raise self._err("illegal instruction")
