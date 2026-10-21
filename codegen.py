"""Code generator: three-address IR -> CVM stack bytecode.

Strategy
  * every variable of a function gets a slot in that function's frame
    (parameters take slots 0..n-1, the rest in order of first appearance)
  * each IR statement becomes a short push / operate / store sequence
  * labels and function entry points are resolved by back-patching
  * a 2-instruction prologue `CALL main; HALT` starts the program
"""
from dataclasses import dataclass

from ir import (Assign, BinOp, Call, Goto, IfNz, IfZ, Label, Print, Return)
from isa import Instr, Op


class CodegenError(Exception):
    """Raised when an error occurs during code generation."""
    pass


BINOP_CODE = {
    "+": Op.ADD, "-": Op.SUB, "*": Op.MUL, "/": Op.DIV, "%": Op.MOD,
    "<": Op.LT, "<=": Op.LE, ">": Op.GT, ">=": Op.GE, "==": Op.EQ, "!=": Op.NE,
}


@dataclass
class Program:
    code: list
    funcs: dict    # address -> function name
    labels: dict   # address -> [label names]


def _vars(ins):
    if isinstance(ins, Assign):
        items = [ins.dst, ins.src]
    elif isinstance(ins, BinOp):
        items = [ins.dst, ins.a, ins.b]
    elif isinstance(ins, (IfZ, IfNz)):
        items = [ins.cond]
    elif isinstance(ins, Print):
        items = [ins.arg]
    elif isinstance(ins, Return):
        items = [ins.arg]
    elif isinstance(ins, Call):
        items = [ins.dst] + list(ins.args)
    else:
        items = []
    return [x for x in items if isinstance(x, str)]


def generate(functions, reg_map=None):
    """Generate CVM stack and register bytecode from a list of IR functions."""
    sigs = {}
    for f in functions:
        sigs[f.name] = len(f.params)
    if "main" not in sigs:
        raise CodegenError("program has no 'main' function")
    if sigs["main"] != 0:
        raise CodegenError("'main' must take no parameters")

    code = [Instr(Op.CALL, 0, 0), Instr(Op.HALT)]   # prologue, patched below
    entry, funcs, labels, call_fixups = {}, {}, {}, []

    for f in functions:
        entry[f.name] = len(code)
        funcs[len(code)] = f.name
        fn_reg_map = None
        if isinstance(reg_map, dict):
            if f.name in reg_map and isinstance(reg_map[f.name], dict):
                fn_reg_map = reg_map[f.name]
            elif not any(isinstance(v, dict) for v in reg_map.values()):
                fn_reg_map = reg_map
        _gen_function(f, code, sigs, call_fixups, labels, reg_map=fn_reg_map)

    for idx, name in call_fixups:
        code[idx] = Instr(Op.CALL, entry[name], code[idx].b)
    code[0] = Instr(Op.CALL, entry["main"], 0)
    return Program(code, funcs, labels)


def _gen_function(fn, code, sigs, call_fixups, labels, reg_map=None):
    slots = {}
    if reg_map:
        for v, r in reg_map.items():
            slots[v] = r

    for p in fn.params:
        if p not in slots:
            next_slot = 8 if reg_map else 0
            while next_slot in slots.values():
                next_slot += 1
            slots[p] = next_slot

    def slot(name):
        if name not in slots:
            next_slot = 8 if reg_map else 0
            while next_slot in slots.values():
                next_slot += 1
            slots[name] = next_slot
        return slots[name]

    for ins in fn.body:            # pre-pass: allocate every variable
        for v in _vars(ins):
            slot(v)

    enter_at = len(code)
    code.append(Instr(Op.ENTER, 0))            # patched with final slot count
    local_labels, jump_fixups = {}, []

    if reg_map:
        for i, p in enumerate(fn.params):
            dst_slot = slot(p)
            if dst_slot != i:
                if dst_slot < 8 and i < 8:
                    code.append(Instr(Op.RLOAD, dst_slot, i))
                elif i < 8:
                    code.append(Instr(Op.RSTORE, i, dst_slot))
                else:
                    code.append(Instr(Op.LOAD, i))
                    code.append(Instr(Op.STORE, dst_slot))

    def push(x):
        if isinstance(x, int):
            if not -2**31 <= x < 2**31:
                raise CodegenError(f"constant {x} does not fit in 32 bits")
            code.append(Instr(Op.PUSH, x))
        else:
            code.append(Instr(Op.LOAD, slots[x]))

    def jump(op, target, line):
        jump_fixups.append((len(code), target, line))
        code.append(Instr(op, 0))

    for ins in fn.body:
        if isinstance(ins, Label):
            if ins.name in local_labels:
                raise CodegenError(f"line {ins.line}: duplicate label '{ins.name}'")
            local_labels[ins.name] = len(code)
        elif isinstance(ins, Assign):
            if reg_map:
                dst = ins.dst
                src = ins.src
                if isinstance(src, int) and dst in reg_map:
                    code.append(Instr(Op.RLOAD_IMM, reg_map[dst], src))
                elif isinstance(src, str) and dst in reg_map and src in reg_map:
                    rd = reg_map[dst]
                    rs = reg_map[src]
                    if rd != rs:
                        code.append(Instr(Op.RLOAD, rd, rs))
                elif isinstance(src, str) and src in reg_map and dst not in reg_map:
                    code.append(Instr(Op.RSTORE, reg_map[src], slot(dst)))
                else:
                    push(ins.src)
                    code.append(Instr(Op.STORE, slot(ins.dst)))
            else:
                push(ins.src)
                code.append(Instr(Op.STORE, slot(ins.dst)))
        elif isinstance(ins, BinOp):
            push(ins.a)
            push(ins.b)
            code.append(Instr(BINOP_CODE[ins.op]))
            code.append(Instr(Op.STORE, slot(ins.dst)))
        elif isinstance(ins, Goto):
            jump(Op.JMP, ins.target, ins.line)
        elif isinstance(ins, IfZ):
            push(ins.cond)
            jump(Op.JZ, ins.target, ins.line)
        elif isinstance(ins, IfNz):
            push(ins.cond)
            jump(Op.JNZ, ins.target, ins.line)
        elif isinstance(ins, Print):
            push(ins.arg)
            code.append(Instr(Op.PRINT))
        elif isinstance(ins, Return):
            if ins.arg is None:
                code.append(Instr(Op.PUSH, 0))
            else:
                push(ins.arg)
            code.append(Instr(Op.RET))
        elif isinstance(ins, Call):
            if ins.func not in sigs:
                raise CodegenError(f"line {ins.line}: call to unknown function '{ins.func}'")
            if len(ins.args) != sigs[ins.func]:
                raise CodegenError(
                    f"line {ins.line}: '{ins.func}' expects {sigs[ins.func]} argument(s), "
                    f"got {len(ins.args)}")
            for a in ins.args:
                push(a)
            call_fixups.append((len(code), ins.func))
            code.append(Instr(Op.CALL, 0, len(ins.args)))
            if ins.dst:
                code.append(Instr(Op.STORE, slot(ins.dst)))
            else:
                code.append(Instr(Op.POP))       # discard unused result
        else:
            raise CodegenError(f"unsupported IR node {ins!r}")

    if not fn.body or not isinstance(fn.body[-1], Return):   # implicit `return 0`
        code.append(Instr(Op.PUSH, 0))
        code.append(Instr(Op.RET))

    for idx, name, line in jump_fixups:
        if name not in local_labels:
            raise CodegenError(f"line {line}: undefined label '{name}' in function '{fn.name}'")
        code[idx] = Instr(code[idx].op, local_labels[name])

    final_slots_count = max(8 if reg_map else 0, max(slots.values()) + 1 if slots else 0)
    code[enter_at] = Instr(Op.ENTER, final_slots_count)
    for name, addr in local_labels.items():
        labels.setdefault(addr, []).append(name)
