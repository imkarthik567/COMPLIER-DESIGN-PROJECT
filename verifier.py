"""Bytecode Verifier for the CVM Stack Machine.

Performs static verification of CVM bytecode before VM execution, similar to the JVM verifier:
1. MAGIC NUMBER CHECK: Confirms bytecode begins with b"CVM1".
2. STRUCTURAL INTEGRITY: Valid opcodes, slot indices within frame bounds, no NaN/sentinel operands,
   exact header instruction count match.
3. CONTROL FLOW SAFETY: Valid jump/call targets, no falling off function ends, function termination with RET,
   dead code warnings after unconditional jumps.
4. STACK DEPTH ANALYSIS: Abstract interpretation simulating operand stack depth, enforcing stack balance
   at join points, preventing stack underflow/overflow, and ensuring the stack is empty upon return.
5. TYPE CONSISTENCY: Lightweight tracking of integers vs function references, ensuring arithmetic
   instructions only operate on integers and call targets are valid function entry points.
6. VERIFICATION RESULT: Structured report (passed, errors, warnings, stats).
"""
from dataclasses import dataclass, field
from enum import Enum
import math
import struct
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from isa import MAGIC, Op, Instr, decode, encode


class Type(str, Enum):
    """Abstract operand types tracked during bytecode verification simulation."""
    INT = "int"
    FUNCREF = "funcref"
    UNKNOWN = "unknown"


class FuncRef:
    """Represents a reference to a function entry point."""
    def __init__(self, target: int):
        self.target = target

    def __repr__(self) -> str:
        return f"FuncRef({self.target})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, FuncRef) and self.target == other.target

    def __hash__(self) -> int:
        return hash(self.target)


@dataclass
class VerificationError:
    instruction_index: int
    code: str
    message: str

    @property
    def index(self) -> int:
        return self.instruction_index


@dataclass
class VerificationWarning:
    instruction_index: int
    code: str
    message: str

    @property
    def index(self) -> int:
        return self.instruction_index


@dataclass
class VerificationResult:
    passed: bool
    errors: List[VerificationError] = field(default_factory=list)
    warnings: List[VerificationWarning] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed

    def format_report(self) -> str:
        """Format verification outcome, error log, and warnings into a readable report."""
        lines = []
        status_str = "PASSED" if self.passed else "FAILED"
        lines.append(f"Bytecode Verification: {status_str}")
        lines.append("-" * 40)
        lines.append(f"  Instructions verified: {self.stats.get('instruction_count', 0)}")
        lines.append(f"  Functions found:       {self.stats.get('functions_found', 0)}")
        lines.append(f"  Max stack depth:       {self.stats.get('stack_depth_max', 0)}")
        lines.append(f"  Errors:                {len(self.errors)}")
        lines.append(f"  Warnings:              {len(self.warnings)}")

        if self.errors:
            lines.append("\nErrors:")
            for err in self.errors:
                lines.append(f"  [inst {err.instruction_index:04d}] [{err.code}] {err.message}")

        if self.warnings:
            lines.append("\nWarnings:")
            for warn in self.warnings:
                lines.append(f"  [inst {warn.instruction_index:04d}] [{warn.code}] {warn.message}")

        return "\n".join(lines)


def print_verification_errors(result: VerificationResult, out=sys.stderr):
    """Print formatted verification failure report."""
    print("Bytecode verification failed with errors:", file=out)
    for err in result.errors:
        print(f"  [error] inst {err.instruction_index:04d}: [{err.code}] {err.message}", file=out)
    for warn in result.warnings:
        print(f"  [warning] inst {warn.instruction_index:04d}: [{warn.code}] {warn.message}", file=out)


class Verifier:
    """Static bytecode verifier for CVM."""

    def __init__(self, max_stack_depth: int = 1000):
        self.max_stack_depth = max_stack_depth

    def verify(self, bytecode: Union[bytes, bytearray, list, Any]) -> VerificationResult:
        """Verify bytecode supplied as raw bytes, a list of Instr, or a Program object."""
        errors: List[VerificationError] = []
        warnings: List[VerificationWarning] = []
        stats: Dict[str, Any] = {
            "stack_depth_max": 0,
            "instruction_count": 0,
            "functions_found": 0,
        }

        # -------------------------------------------------------------
        # 1. MAGIC NUMBER CHECK & BINARY STRUCTURAL INTEGRITY
        # -------------------------------------------------------------
        code: List[Instr] = []
        if isinstance(bytecode, (bytes, bytearray)):
            if len(bytecode) < 4 or bytecode[:4] != MAGIC:
                errors.append(
                    VerificationError(
                        instruction_index=0,
                        code="BAD_MAGIC",
                        message=f"file does not start with CVM1 magic bytes (got {bytecode[:4]!r})",
                    )
                )
                return VerificationResult(passed=False, errors=errors, warnings=warnings, stats=stats)

            if len(bytecode) < 8:
                errors.append(
                    VerificationError(
                        instruction_index=0,
                        code="TRUNCATED_HEADER",
                        message="bytecode file truncated (less than 8 bytes)",
                    )
                )
                return VerificationResult(passed=False, errors=errors, warnings=warnings, stats=stats)

            (declared_count,) = struct.unpack_from("<I", bytecode, 4)
            expected_size = 8 + declared_count * 9
            if len(bytecode) != expected_size:
                errors.append(
                    VerificationError(
                        instruction_index=0,
                        code="COUNT_MISMATCH",
                        message=(
                            f"instruction count mismatch: header declares {declared_count} instructions "
                            f"({expected_size} bytes), but file contains {len(bytecode)} bytes"
                        ),
                    )
                )
                return VerificationResult(passed=False, errors=errors, warnings=warnings, stats=stats)

            try:
                code = decode(bytecode)
            except ValueError as e:
                errors.append(VerificationError(instruction_index=0, code="DECODE_ERROR", message=str(e)))
                return VerificationResult(passed=False, errors=errors, warnings=warnings, stats=stats)

        elif hasattr(bytecode, "code"):  # e.g. Program dataclass
            code = bytecode.code
        elif isinstance(bytecode, list):
            code = bytecode
        else:
            errors.append(
                VerificationError(
                    instruction_index=0,
                    code="INVALID_INPUT",
                    message=f"unsupported bytecode input type: {type(bytecode).__name__}",
                )
            )
            return VerificationResult(passed=False, errors=errors, warnings=warnings, stats=stats)

        stats["instruction_count"] = len(code)
        if not code:
            errors.append(VerificationError(instruction_index=0, code="EMPTY_BYTECODE", message="bytecode is empty"))
            return VerificationResult(passed=False, errors=errors, warnings=warnings, stats=stats)

        # -------------------------------------------------------------
        # 2. STRUCTURAL INTEGRITY
        # -------------------------------------------------------------
        valid_opcodes = set(Op)
        for i, ins in enumerate(code):
            if not isinstance(ins, Instr) or ins.op not in valid_opcodes:
                errors.append(
                    VerificationError(
                        instruction_index=i,
                        code="INVALID_OPCODE",
                        message=f"invalid opcode {getattr(ins, 'op', ins)!r} at instruction {i}",
                    )
                )
                continue

            # Check operands for garbage (NaN, Inf, sentinel, range)
            for op_name, val in (("a", ins.a), ("b", ins.b)):
                if isinstance(val, float):
                    if math.isnan(val) or math.isinf(val):
                        errors.append(
                            VerificationError(
                                instruction_index=i,
                                code="GARBAGE_OPERAND",
                                message=f"operand '{op_name}' is NaN/Inf at instruction {i}",
                            )
                        )
                elif isinstance(val, int):
                    if not (-2**31 <= val < 2**31):
                        errors.append(
                            VerificationError(
                                instruction_index=i,
                                code="OPERAND_OUT_OF_RANGE",
                                message=f"operand '{op_name}' ({val}) out of 32-bit signed range at instruction {i}",
                            )
                        )
                elif isinstance(val, FuncRef):
                    pass  # valid function reference
                else:
                    errors.append(
                        VerificationError(
                            instruction_index=i,
                            code="GARBAGE_OPERAND",
                            message=f"operand '{op_name}' has unrecognized type {type(val).__name__} at instruction {i}",
                        )
                    )

        # Discover function entry points (Op.ENTER)
        func_entries: List[int] = [i for i, ins in enumerate(code) if isinstance(ins, Instr) and ins.op == Op.ENTER]
        stats["functions_found"] = len(func_entries)

        # Map each instruction to its containing function and frame size
        # If no ENTER instructions exist, entire code is a single frame with frame_size = 0
        ins_frame_size: Dict[int, int] = {}
        ins_func_entry: Dict[int, Optional[int]] = {}

        if not func_entries:
            for i in range(len(code)):
                ins_frame_size[i] = 0
                ins_func_entry[i] = None
        else:
            # Top-level prologue before first function (e.g. CALL main; HALT)
            first_fe = func_entries[0]
            for i in range(first_fe):
                ins_frame_size[i] = 0
                ins_func_entry[i] = None

            for k, fe in enumerate(func_entries):
                f_end = func_entries[k + 1] if k + 1 < len(func_entries) else len(code)
                enter_ins = code[fe]
                frame_sz = enter_ins.a if isinstance(enter_ins.a, int) else 0
                if frame_sz < 0:
                    errors.append(
                        VerificationError(
                            instruction_index=fe,
                            code="NEGATIVE_FRAME_SIZE",
                            message=f"negative frame size {frame_sz} at instruction {fe}",
                        )
                    )
                for i in range(fe, f_end):
                    ins_frame_size[i] = frame_sz
                    ins_func_entry[i] = fe

        # Verify slot bounds (0 <= slot < frame_size) for LOAD and STORE
        for i, ins in enumerate(code):
            if isinstance(ins, Instr) and ins.op in (Op.LOAD, Op.STORE):
                slot = ins.a
                f_size = ins_frame_size.get(i, 0)
                if not isinstance(slot, int) or slot < 0 or slot >= f_size:
                    errors.append(
                        VerificationError(
                            instruction_index=i,
                            code="BAD_SLOT",
                            message=(
                                f"variable slot index {slot} out of bounds for frame size {f_size} "
                                f"at instruction {i} ({ins.op.name})"
                            ),
                        )
                    )

        # -------------------------------------------------------------
        # 3. CONTROL FLOW SAFETY: TARGETS & FUNCTION ENCAPSULATION
        # -------------------------------------------------------------
        func_entry_set = set(func_entries)

        for i, ins in enumerate(code):
            if not isinstance(ins, Instr):
                continue

            # Jumps: JMP, JZ, JNZ
            if ins.op in (Op.JMP, Op.JZ, Op.JNZ):
                target = ins.a
                if not isinstance(target, int):
                    errors.append(
                        VerificationError(
                            instruction_index=i,
                            code="BAD_JUMP_TARGET",
                            message=f"non-integer jump target {target!r} at instruction {i}",
                        )
                    )
                elif not (0 <= target < len(code)):
                    errors.append(
                        VerificationError(
                            instruction_index=i,
                            code="BAD_JUMP_TARGET",
                            message=f"jump target {target} out of bounds [0, {len(code)}) at instruction {i}",
                        )
                    )
                else:
                    # Jumping directly to an ENTER instruction of another function is illegal
                    if target in func_entry_set and target != ins_func_entry.get(i):
                        errors.append(
                            VerificationError(
                                instruction_index=i,
                                code="ILLEGAL_JUMP",
                                message=f"jump at instruction {i} directly targets function entry {target}",
                            )
                        )

            # Calls: CALL target, n_args
            elif ins.op == Op.CALL:
                target = ins.a.target if isinstance(ins.a, FuncRef) else ins.a
                n_args = ins.b
                if not isinstance(target, int) or not (0 <= target < len(code)):
                    errors.append(
                        VerificationError(
                            instruction_index=i,
                            code="BAD_CALL_TARGET",
                            message=f"call target {target!r} out of bounds [0, {len(code)}) at instruction {i}",
                        )
                    )
                else:
                    target_ins = code[target]
                    if not isinstance(target_ins, Instr) or target_ins.op != Op.ENTER:
                        errors.append(
                            VerificationError(
                                instruction_index=i,
                                code="INVALID_FUNCTION_ENTRY",
                                message=(
                                    f"call target {target} is not a function entry point "
                                    f"(opcode is {getattr(target_ins, 'op', target_ins)}, expected ENTER)"
                                ),
                            )
                        )
                    else:
                        target_frame_sz = target_ins.a
                        if isinstance(n_args, int) and isinstance(target_frame_sz, int):
                            if n_args < 0:
                                errors.append(
                                    VerificationError(
                                        instruction_index=i,
                                        code="NEGATIVE_ARG_COUNT",
                                        message=f"call at instruction {i} has negative argument count {n_args}",
                                    )
                                )
                            elif n_args > target_frame_sz:
                                errors.append(
                                    VerificationError(
                                        instruction_index=i,
                                        code="TOO_MANY_ARGS",
                                        message=(
                                            f"call at instruction {i} passes {n_args} arguments, but target "
                                            f"function frame size is only {target_frame_sz}"
                                        ),
                                    )
                                )

        # If structural/target errors exist that prevent interpretation, abort early
        fatal_error_codes = {"INVALID_OPCODE", "BAD_JUMP_TARGET", "BAD_CALL_TARGET"}
        if any(e.code in fatal_error_codes for e in errors):
            return VerificationResult(passed=False, errors=errors, warnings=warnings, stats=stats)

        # -------------------------------------------------------------
        # 4 & 5. ABSTRACT INTERPRETATION: STACK DEPTH & TYPES
        # -------------------------------------------------------------
        # Routines to analyze:
        # - Top-level execution starting at pc = 0
        # - Each function starting at pc = fe
        routine_entries = [0] + [fe for fe in func_entries if fe != 0]

        visited_pcs: Set[int] = set()
        stack_depth_at_pc: Dict[int, int] = {}
        stack_types_at_pc: Dict[int, List[Type]] = {}
        slot_types_at_pc: Dict[int, Dict[int, Type]] = {}
        max_seen_depth = 0

        for r_entry in routine_entries:
            f_size = ins_frame_size.get(r_entry, 0)
            initial_slots = {s: Type.INT for s in range(f_size)}
            initial_state = (r_entry, 0, [], initial_slots)
            worklist = [initial_state]

            # Track which instructions in this routine terminate with RET / HALT
            routine_visited: Set[int] = set()
            routine_exits: Set[int] = set()

            while worklist:
                pc, depth, stack_types, slot_types = worklist.pop()
                if pc >= len(code):
                    errors.append(
                        VerificationError(
                            instruction_index=pc - 1,
                            code="FALL_OFF_END",
                            message=f"control falls off end of code from instruction {pc - 1}",
                        )
                    )
                    continue

                visited_pcs.add(pc)
                routine_visited.add(pc)
                max_seen_depth = max(max_seen_depth, depth)

                if depth > self.max_stack_depth:
                    errors.append(
                        VerificationError(
                            instruction_index=pc,
                            code="STACK_OVERFLOW",
                            message=f"stack depth {depth} exceeds limit {self.max_stack_depth} at instruction {pc}",
                        )
                    )
                    continue

                # Merge point verification: same stack depth required across all incoming paths
                if pc in stack_depth_at_pc:
                    existing_depth = stack_depth_at_pc[pc]
                    if existing_depth != depth:
                        errors.append(
                            VerificationError(
                                instruction_index=pc,
                                code="STACK_DEPTH_MISMATCH",
                                message=(
                                    f"stack depth mismatch at merge point instruction {pc}: "
                                    f"incoming depth is {depth}, but previous path left depth {existing_depth}"
                                ),
                            )
                        )
                        continue
                    if stack_types_at_pc.get(pc) == stack_types and slot_types_at_pc.get(pc) == slot_types:
                        continue
                    stack_types_at_pc[pc] = list(stack_types)
                    slot_types_at_pc[pc] = dict(slot_types)
                else:
                    stack_depth_at_pc[pc] = depth
                    stack_types_at_pc[pc] = list(stack_types)
                    slot_types_at_pc[pc] = dict(slot_types)

                ins = code[pc]
                op = ins.op

                # Simulate stack depth and types for each instruction
                next_depth = depth
                next_stack = list(stack_types)
                next_slots = dict(slot_types)
                successors: List[int] = []

                if op == Op.PUSH:
                    val_type = Type.FUNCREF if isinstance(ins.a, FuncRef) else Type.INT
                    next_depth = depth + 1
                    next_stack.append(val_type)
                    successors.append(pc + 1)

                elif op == Op.POP:
                    if depth < 1:
                        errors.append(
                            VerificationError(
                                instruction_index=pc,
                                code="STACK_UNDERFLOW",
                                message=f"stack underflow at instruction {pc} (POP requires 1 operand, depth is {depth})",
                            )
                        )
                        continue
                    next_depth = depth - 1
                    next_stack.pop()
                    successors.append(pc + 1)

                elif op == Op.DUP:
                    if depth < 1:
                        errors.append(
                            VerificationError(
                                instruction_index=pc,
                                code="STACK_UNDERFLOW",
                                message=f"stack underflow at instruction {pc} (DUP requires 1 operand, depth is {depth})",
                            )
                        )
                        continue
                    top_t = next_stack[-1]
                    next_depth = depth + 1
                    next_stack.append(top_t)
                    successors.append(pc + 1)

                elif op == Op.LOAD:
                    slot = ins.a
                    val_t = next_slots.get(slot, Type.INT)
                    next_depth = depth + 1
                    next_stack.append(val_t)
                    successors.append(pc + 1)

                elif op == Op.STORE:
                    if depth < 1:
                        errors.append(
                            VerificationError(
                                instruction_index=pc,
                                code="STACK_UNDERFLOW",
                                message=f"stack underflow at instruction {pc} (STORE requires 1 operand, depth is {depth})",
                            )
                        )
                        continue
                    val_t = next_stack.pop()
                    next_depth = depth - 1
                    next_slots[ins.a] = val_t
                    successors.append(pc + 1)

                elif op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD, Op.LT, Op.LE, Op.GT, Op.GE, Op.EQ, Op.NE):
                    if depth < 2:
                        errors.append(
                            VerificationError(
                                instruction_index=pc,
                                code="STACK_UNDERFLOW",
                                message=(
                                    f"stack underflow at instruction {pc} ({op.name} requires 2 operands, "
                                    f"depth is {depth})"
                                ),
                            )
                        )
                        continue
                    tb = next_stack.pop()
                    ta = next_stack.pop()
                    if ta == Type.FUNCREF or tb == Type.FUNCREF:
                        errors.append(
                            VerificationError(
                                instruction_index=pc,
                                code="TYPE_MISMATCH",
                                message=(
                                    f"type mismatch at instruction {pc}: arithmetic/comparison instruction "
                                    f"{op.name} cannot operate on function reference (types: {ta.value}, {tb.value})"
                                ),
                            )
                        )
                    next_depth = depth - 1
                    next_stack.append(Type.INT)
                    successors.append(pc + 1)

                elif op == Op.PRINT:
                    if depth < 1:
                        errors.append(
                            VerificationError(
                                instruction_index=pc,
                                code="STACK_UNDERFLOW",
                                message=f"stack underflow at instruction {pc} (PRINT requires 1 operand, depth is {depth})",
                            )
                        )
                        continue
                    next_stack.pop()
                    next_depth = depth - 1
                    successors.append(pc + 1)

                elif op == Op.ENTER:
                    successors.append(pc + 1)

                elif op == Op.CALL:
                    n_args = ins.b
                    if depth < n_args:
                        errors.append(
                            VerificationError(
                                instruction_index=pc,
                                code="STACK_UNDERFLOW",
                                message=(
                                    f"stack underflow at instruction {pc} (CALL expects {n_args} arguments, "
                                    f"depth is {depth})"
                                ),
                            )
                        )
                        continue
                    if n_args > 0:
                        next_stack = next_stack[:-n_args]
                    next_stack.append(Type.INT)
                    next_depth = depth - n_args + 1
                    successors.append(pc + 1)

                elif op == Op.JMP:
                    target = ins.a
                    if 0 <= target < len(code):
                        successors.append(target)

                elif op in (Op.JZ, Op.JNZ):
                    if depth < 1:
                        errors.append(
                            VerificationError(
                                instruction_index=pc,
                                code="STACK_UNDERFLOW",
                                message=f"stack underflow at instruction {pc} ({op.name} requires 1 operand, depth is {depth})",
                            )
                        )
                        continue
                    next_stack.pop()
                    next_depth = depth - 1
                    target = ins.a
                    if 0 <= target < len(code):
                        successors.append(target)
                    successors.append(pc + 1)

                elif op == Op.RET:
                    routine_exits.add(pc)
                    if depth < 1:
                        errors.append(
                            VerificationError(
                                instruction_index=pc,
                                code="STACK_UNDERFLOW",
                                message=(
                                    f"stack underflow at RET instruction {pc}: "
                                    f"requires return value on stack, but stack depth is 0"
                                ),
                            )
                        )
                    elif depth > 1:
                        errors.append(
                            VerificationError(
                                instruction_index=pc,
                                code="STACK_NOT_EMPTY",
                                message=(
                                    f"stack must be empty at RET instruction {pc}: "
                                    f"expected 1 return value on stack, found depth {depth} (leftover items)"
                                ),
                            )
                        )
                    # RET terminates current control path

                elif op == Op.HALT:
                    routine_exits.add(pc)
                    # HALT terminates current control path

                # Enqueue successors
                for succ in successors:
                    # Check boundary: cannot fall through into another function's ENTER
                    if succ in func_entry_set and succ != ins_func_entry.get(pc):
                        errors.append(
                            VerificationError(
                                instruction_index=pc,
                                code="FALL_INTO_FUNCTION",
                                message=f"instruction {pc} falls through into function entry {succ} without RET",
                            )
                        )
                    else:
                        worklist.append((succ, next_depth, next_stack, next_slots))

            # Verify that every function has a valid return path
            if r_entry in func_entry_set:
                has_ret = any(code[exit_pc].op == Op.RET for exit_pc in routine_exits if exit_pc < len(code))
                if not has_ret:
                    errors.append(
                        VerificationError(
                            instruction_index=r_entry,
                            code="UNTERMINATED_FUNCTION",
                            message=f"function starting at instruction {r_entry} has no valid RET path",
                        )
                    )

        stats["stack_depth_max"] = max_seen_depth

        # -------------------------------------------------------------
        # 6. DEAD CODE / UNREACHABLE INSTRUCTION WARNINGS
        # -------------------------------------------------------------
        # Identify instructions never visited by any path from routine entries
        for i, ins in enumerate(code):
            if i not in visited_pcs:
                # Preceded by unconditional jump / ret / halt
                if i > 0 and code[i - 1].op in (Op.JMP, Op.RET, Op.HALT):
                    warnings.append(
                        VerificationWarning(
                            instruction_index=i,
                            code="DEAD_CODE",
                            message=f"unreachable dead code after unconditional {code[i-1].op.name} at instruction {i}",
                        )
                    )
                else:
                    warnings.append(
                        VerificationWarning(
                            instruction_index=i,
                            code="UNREACHABLE_CODE",
                            message=f"instruction {i} ({ins.op.name}) is unreachable",
                        )
                    )

        passed = len(errors) == 0
        return VerificationResult(
            passed=passed,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )
