from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count
import math

from isa import Op, format_instr, REDUCTION_NAMES


def _parallel_worker(payload):
    """Run one isolated compiled worker function in a child process."""
    code, entry, value, max_steps, max_depth = payload
    prefix_size = 3  # PUSH argument, CALL worker, HALT
    shifted = []
    for ins in code:
        if ins.op in (Op.JMP, Op.JZ, Op.JNZ, Op.CALL, Op.PAR_SUM, Op.PAR_REDUCE):
            shifted.append(ins._replace(a=ins.a + prefix_size))
        else:
            shifted.append(ins)
    task_code = [
        type(code[0])(Op.PUSH, value),
        type(code[0])(Op.CALL, entry + prefix_size, 1),
        type(code[0])(Op.HALT),
        *shifted,
    ]
    vm = VM(task_code, max_steps=max_steps, max_depth=max_depth, out=lambda _value: None)
    vm.run()
    if not vm.stack:
        raise VMError("parallel worker returned no value")
    return vm.stack[-1], vm.steps


def _parallel_worker_batch(payload):
    code, entry, values, max_steps, max_depth = payload
    return [_parallel_worker((code, entry, value, max_steps, max_depth)) for value in values]


def _reduce_parallel(kind, inputs, values):
    if kind == "product": return math.prod(values)
    if kind == "count": return sum(value != 0 for value in values)
    if kind == "any": return int(any(values))
    if kind == "all": return int(all(values))
    if kind == "bit_and":
        result = -1
        for value in values: result &= value
        return result
    if kind == "bit_or":
        result = 0
        for value in values: result |= value
        return result
    if kind == "bit_xor":
        result = 0
        for value in values: result ^= value
        return result
    if kind == "gcd": return math.gcd(*values)
    if kind == "lcm": return math.lcm(*values)
    if kind == "average": return sum(values) // len(values)
    if kind == "argmin":
        return min(zip(inputs, values), key=lambda pair: (pair[1], pair[0]))[0]
    if kind == "argmax":
        return min(zip(inputs, values), key=lambda pair: (-pair[1], pair[0]))[0]
    if kind == "min": return min(values)
    if kind == "max": return max(values)
    raise VMError(f"unknown parallel reduction '{kind}'")


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
        self.parallel_tasks = 0
        self.parallel_workers = 0
        self.parallel_worker_steps = 0

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
            elif op in (Op.FLOOR_DIV, Op.PY_MOD):
                b = self._pop()
                a = self._pop()
                if b == 0:
                    raise self._err("division by zero")
                self.stack.append(a // b if op == Op.FLOOR_DIV else a % b)
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
            elif op == Op.PAR_SUM:
                stop = self._pop()
                start = self._pop()
                if type(start) is not int or type(stop) is not int:
                    raise self._err("parallel_sum bounds must be integers")
                count = max(0, stop - start)
                if count > 100_000:
                    raise self._err("parallel_sum range exceeds 100000 tasks")
                if count == 0:
                    self.stack.append(0)
                else:
                    workers = min(count, cpu_count() or 1)
                    self.parallel_tasks += count
                    self.parallel_workers = max(self.parallel_workers, workers)
                    chunk = max(1, (count + workers * 4 - 1) // (workers * 4))
                    ranges = [range(left, min(left + chunk, stop)) for left in range(start, stop, chunk)]
                    payloads = ((code, ins.a, tuple(values), self.max_steps, self.max_depth) for values in ranges)
                    with ProcessPoolExecutor(max_workers=workers) as pool:
                        results = [item for batch in pool.map(_parallel_worker_batch, payloads) for item in batch]
                    self.stack.append(sum(result for result, _steps in results))
                    self.parallel_worker_steps += sum(steps for _result, steps in results)
            elif op == Op.PAR_REDUCE:
                stop = self._pop()
                start = self._pop()
                if type(start) is not int or type(stop) is not int:
                    raise self._err("parallel reduction bounds must be integers")
                count = max(0, stop - start)
                if count > 100_000:
                    raise self._err("parallel reduction range exceeds 100000 tasks")
                kind = REDUCTION_NAMES.get(ins.b)
                if kind is None:
                    raise self._err(f"unknown parallel reduction mode {ins.b}")
                if count == 0:
                    identities = {"product": 1, "count": 0, "any": 0, "all": 1,
                                  "bit_and": -1, "bit_or": 0, "bit_xor": 0,
                                  "gcd": 0, "lcm": 1, "sum": 0}
                    if kind not in identities:
                        raise self._err(f"parallel_{kind} cannot reduce an empty range")
                    self.stack.append(identities[kind])
                    continue
                workers = min(count, cpu_count() or 1)
                self.parallel_tasks += count
                self.parallel_workers = max(self.parallel_workers, workers)
                chunk = max(1, (count + workers * 4 - 1) // (workers * 4))
                ranges = [range(left, min(left + chunk, stop)) for left in range(start, stop, chunk)]
                payloads = ((code, ins.a, tuple(values), self.max_steps, self.max_depth) for values in ranges)
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    results = [item for batch in pool.map(_parallel_worker_batch, payloads) for item in batch]
                values = [result for result, _steps in results]
                self.stack.append(_reduce_parallel(kind, range(start, stop), values))
                self.parallel_worker_steps += sum(steps for _result, steps in results)
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
