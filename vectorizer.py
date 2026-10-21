"""CVM SIMD Loop Vectorizer.

Phase 7 of CVM optimizing compiler backend.
Detects and transforms loops that can be vectorized with SIMD vector width W = 4
(processing multiple array elements per step instead of one).

Runs BEFORE Phase 5 (register allocation) and AFTER Phase 3 (loop optimization).

Provides:
- ArrayAccess: Structured representation of array loads/stores with affine index analysis
- gcd_dependence_test: Exact GCD-based Diophantine dependence test for array accesses
- check_vectorizability: Verifies the 6 required vectorizability conditions
- vectorize_loop: Transforms scalar loop into vector loop (width 4) + scalar remainder loop
- vectorize_function / vectorize_program: Pipeline drivers
- LoopVectorizationResult & VectorizationStats: Reporting and statistics
- Transparent runtime integration for codegen, dataflow, and verifier
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import codegen
from optimizer import BasicBlock, build_cfg, cfg_to_function
import dataflow
from ir import (
    Assign,
    BinOp,
    Call,
    Function,
    Goto,
    IfNz,
    IfZ,
    Label,
    LoadIndexed,
    NewArray,
    Operand,
    Print,
    Return,
    StoreIndexed,
    VAdd,
    VDiv,
    VLoad,
    VMul,
    VStore,
    VSub,
)
from isa import Instr, Op
from loops import NaturalLoop, find_natural_loops, find_primary_induction_var
from optimizer import BasicBlock, build_cfg, cfg_to_function
import loops
import regalloc
import scheduler
import verifier


# =====================================================================
# STEP 1: Array Access Representation and Dependence Analysis
# =====================================================================

@dataclass
class ArrayAccess:
    """Represents an array load or store instruction with affine index metadata."""
    array: str
    is_store: bool
    offset: int
    ins: Any
    coeff: int = 1
    index_expr: Optional[str] = None


def gcd_dependence_test(acc1: ArrayAccess, acc2: ArrayAccess) -> Tuple[bool, Optional[int]]:
    """Perform GCD test for loop-carried dependence between two array accesses.

    For accesses A[a1 * i + c1] and A[a2 * i + c2]:
      a1 * i1 - a2 * i2 = c2 - c1
    Let g = gcd(a1, a2).
    If (c2 - c1) % g != 0:
      No integer solutions -> strictly independent.
    If a1 == a2 == 1:
      distance = c1 - c2.
      If distance == 0: same iteration (intra-iteration, no loop-carried hazard).
      If distance != 0: cross-iteration dependence exists.

    Returns:
        (has_cross_iteration_dependence, distance)
    """
    if acc1.array != acc2.array:
        return False, None
    if not acc1.is_store and not acc2.is_store:
        return False, None  # Read-After-Read has no hazard

    a1, c1 = acc1.coeff, acc1.offset
    a2, c2 = acc2.coeff, acc2.offset

    g = math.gcd(a1, a2)
    diff = c2 - c1
    if diff % g != 0:
        # GCD test passes: no integer solutions exist -> independent
        return False, None

    if a1 == a2 == 1:
        distance = c1 - c2
        if distance == 0:
            return False, 0
        return True, distance

    # General Diophantine solution exists
    return True, None


def _extract_array_access(
    ins: Any, iv: str, loop_defs: Dict[str, Any]
) -> Optional[Tuple[bool, Optional[ArrayAccess], str]]:
    """Extract ArrayAccess from LoadIndexed or StoreIndexed instruction.

    Returns:
        (is_array_access, access_obj_or_None, failure_reason)
    """
    if not isinstance(ins, (LoadIndexed, StoreIndexed)):
        return False, None, ""

    array = ins.array
    is_store = isinstance(ins, StoreIndexed)
    idx = ins.index

    # 1. Direct induction variable index: a[i]
    if idx == iv:
        return True, ArrayAccess(array=array, is_store=is_store, offset=0, ins=ins, coeff=1), ""

    # 2. Constant integer index: a[0] -> not of form a[i + constant]
    if isinstance(idx, int):
        return True, None, f"array index {idx} is a constant, not of form a[{iv} + constant]"

    # 3. Variable index: check if derived from iv
    if isinstance(idx, str):
        if idx in loop_defs:
            def_ins = loop_defs[idx]
            if isinstance(def_ins, Assign):
                if def_ins.src == iv:
                    return True, ArrayAccess(array=array, is_store=is_store, offset=0, ins=ins, coeff=1), ""
            elif isinstance(def_ins, BinOp):
                if def_ins.op == "+" and def_ins.a == iv and isinstance(def_ins.b, int):
                    return True, ArrayAccess(array=array, is_store=is_store, offset=def_ins.b, ins=ins, coeff=1), ""
                elif def_ins.op == "+" and def_ins.b == iv and isinstance(def_ins.a, int):
                    return True, ArrayAccess(array=array, is_store=is_store, offset=def_ins.a, ins=ins, coeff=1), ""
                elif def_ins.op == "-" and def_ins.a == iv and isinstance(def_ins.b, int):
                    return True, ArrayAccess(array=array, is_store=is_store, offset=-def_ins.b, ins=ins, coeff=1), ""
                elif def_ins.op == "*" and (def_ins.a == iv or def_ins.b == iv):
                    k = def_ins.b if def_ins.a == iv else def_ins.a
                    return True, None, f"scaled array index with coefficient {k} is not stride 1"

        return True, None, f"array index '{idx}' cannot be proven of form a[{iv} + constant]"

    return False, None, f"unsupported array index {idx!r}"


# =====================================================================
# STEP 2: Vectorizability Check
# =====================================================================

@dataclass
class LoopVectorizationResult:
    """Diagnostic outcome and parameters for an analyzed loop candidate."""
    header: str
    vectorized: bool
    reason: str
    speedup: float = 1.0
    width: int = 4
    trip_count: Optional[int] = None


@dataclass
class VectorizationStats:
    """Aggregated vectorization statistics across all loops in a function or program."""
    loops_analyzed: int = 0
    loops_vectorized: int = 0
    details: List[LoopVectorizationResult] = field(default_factory=list)

    def format_report(self) -> str:
        """Format vectorization statistics into a readable diagnostic summary."""
        lines = [
            "=== VECTORIZATION REPORT ===",
            f"  Loops analyzed:              {self.loops_analyzed}",
            f"  Loops vectorized:            {self.loops_vectorized}",
        ]
        if self.details:
            lines.append("  Details:")
            for d in self.details:
                status = "VECTORIZED" if d.vectorized else "NOT VECTORIZED"
                speedup_str = f"{d.speedup:.2f}x" if d.vectorized else "1.00x"
                lines.append(f"    - Loop '{d.header}': {status}")
                lines.append(f"        Speedup: {speedup_str}")
                lines.append(f"        Reason:  {d.reason}")
        return "\n".join(lines)


def check_vectorizability(
    loop: NaturalLoop, cfg: Dict[str, BasicBlock]
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Verify all 6 vectorizability conditions for loop.

    Conditions:
      (a) Single induction variable with constant stride 1
      (b) All array accesses have the form a[i + constant]
      (c) No cross-iteration dependence exists (GCD test passes for all pairs)
      (d) No function calls inside the loop body
      (e) No early exits (no conditional branches except the loop condition itself)
      (f) The loop body is straight-line code (single basic block body)

    Returns:
        (is_vectorizable, reason_or_status, context_dict)
    """
    # -----------------------------------------------------------------
    # (d) No function calls inside the loop body
    # -----------------------------------------------------------------
    for node in loop.body:
        if node in cfg:
            for ins in cfg[node].instrs:
                if isinstance(ins, Call):
                    return False, f"loop body contains function call '{ins.func}'", None

    # -----------------------------------------------------------------
    # (f) The loop body is straight-line code (single basic block)
    # -----------------------------------------------------------------
    # Allowed structures:
    # 1. len(loop.body) == 1: Single block containing loop test + body
    # 2. len(loop.body) == 2: Header block (test) + Body block (straight-line body)
    if len(loop.body) > 2:
        return False, f"loop body contains multiple basic blocks ({len(loop.body)})", None

    if len(loop.body) == 2:
        body_node = next(n for n in loop.body if n != loop.header)
        body_block = cfg.get(body_node)
        if body_block:
            # Body block must end in an unconditional goto to header
            if not body_block.instrs or not isinstance(body_block.instrs[-1], Goto):
                return False, "loop body does not end in unconditional jump to header", None
            if body_block.instrs[-1].target != loop.header:
                return False, "loop body jump does not target loop header", None
            # No internal branches inside body block
            for ins in body_block.instrs[:-1]:
                if isinstance(ins, (Goto, IfZ, IfNz)):
                    return False, "loop body contains internal branch", None

    # -----------------------------------------------------------------
    # (e) No early exits (only 1 exit block with single exit branch)
    # -----------------------------------------------------------------
    if len(loop.exit_blocks) != 1:
        return False, f"loop has multiple or zero exit blocks ({len(loop.exit_blocks)})", None

    exit_block_lbl = next(iter(loop.exit_blocks))
    if exit_block_lbl not in cfg:
        return False, "exit block missing from CFG", None

    exit_block = cfg[exit_block_lbl]
    exit_branches = [ins for ins in exit_block.instrs if isinstance(ins, (IfZ, IfNz))]
    if len(exit_branches) != 1:
        return False, f"exit block contains {len(exit_branches)} conditional branches (expected 1)", None

    branch_ins = exit_branches[0]

    # -----------------------------------------------------------------
    # (a) Single induction variable with constant stride 1
    # -----------------------------------------------------------------
    iv_info = find_primary_induction_var(loop, cfg)
    if not iv_info:
        return False, "no primary induction variable found", None

    iv_var, step, inc_block, inc_idx, inc_ins = iv_info
    if step != 1:
        return False, f"induction variable '{iv_var}' has stride {step} != 1", None

    # -----------------------------------------------------------------
    # Extract loop definitions and loop bound
    # -----------------------------------------------------------------
    loop_defs: Dict[str, Any] = {}
    for node in loop.body:
        if node in cfg:
            for ins in cfg[node].instrs:
                if isinstance(ins, (Assign, BinOp)):
                    loop_defs[ins.dst] = ins

    # Find the comparison defining branch condition
    cond_var = branch_ins.cond
    if not isinstance(cond_var, str) or cond_var not in loop_defs:
        return False, f"branch condition '{cond_var}' is not defined in loop", None

    cmp_ins = loop_defs[cond_var]
    if not isinstance(cmp_ins, BinOp):
        return False, f"branch condition '{cond_var}' is not a binary comparison", None

    # Canonicalize comparison with iv on left
    swap_ops = {">": "<", ">=": "<=", "<": ">", "<=": ">=", "==": "==", "!=": "!="}
    if cmp_ins.a == iv_var:
        comp_op = cmp_ins.op
        limit = cmp_ins.b
    elif cmp_ins.b == iv_var and cmp_ins.op in swap_ops:
        comp_op = swap_ops[cmp_ins.op]
        limit = cmp_ins.a
    else:
        return False, f"comparison '{cmp_ins.dst}' does not compare induction variable '{iv_var}'", None

    if comp_op not in ("<", "<=", ">", ">="):
        return False, f"unsupported loop comparison operator '{comp_op}'", None

    # Determine exit target
    if branch_ins.target in loop.body:
        # Branch stays in loop; exit is the fallthrough
        exit_target = next(s for s in exit_block.succs if s not in loop.body)
        branch_exits_on_target = False
    else:
        exit_target = branch_ins.target
        branch_exits_on_target = True

    # -----------------------------------------------------------------
    # (b) All array accesses have the form a[i + constant]
    # -----------------------------------------------------------------
    array_accesses: List[ArrayAccess] = []
    has_array_access = False

    for node in loop.body:
        if node in cfg:
            for ins in cfg[node].instrs:
                is_arr, acc, reason = _extract_array_access(ins, iv_var, loop_defs)
                if is_arr:
                    has_array_access = True
                    if acc is None:
                        return False, reason, None
                    array_accesses.append(acc)

    if not has_array_access:
        return False, "no array accesses found in loop", None

    # -----------------------------------------------------------------
    # (c) No cross-iteration dependence exists (GCD test)
    # -----------------------------------------------------------------
    for i in range(len(array_accesses)):
        for j in range(i + 1, len(array_accesses)):
            acc1 = array_accesses[i]
            acc2 = array_accesses[j]
            has_dep, dist = gcd_dependence_test(acc1, acc2)
            if has_dep:
                return False, f"cross-iteration dependence detected on array '{acc1.array}' with distance {dist}", None

    context = {
        "iv": iv_var,
        "step": step,
        "inc_block": inc_block,
        "inc_idx": inc_idx,
        "branch_ins": branch_ins,
        "branch_exits_on_target": branch_exits_on_target,
        "cmp_ins": cmp_ins,
        "comp_op": comp_op,
        "limit": limit,
        "exit_target": exit_target,
        "array_accesses": array_accesses,
    }
    return True, "vectorizable with SIMD width 4", context


# =====================================================================
# STEP 3: Vector IR Generation and Loop Transformation
# =====================================================================

def vectorize_loop(
    loop: NaturalLoop,
    cfg: Dict[str, BasicBlock],
    fn_name: str,
    context: Dict[str, Any],
) -> bool:
    """Transform scalar loop into vector loop (width 4) + scalar remainder loop."""
    iv = context["iv"]
    limit = context["limit"]
    comp_op = context["comp_op"]
    exit_target = context["exit_target"]
    branch_ins = context["branch_ins"]
    branch_exits_on_target = context["branch_exits_on_target"]
    cmp_ins = context["cmp_ins"]

    header_lbl = loop.header
    header_block = cfg[header_lbl]
    outside_preds = [p for p in header_block.preds if p not in loop.body]

    # Extract initial value of induction variable if available
    iv_init = 0
    for pred_lbl in outside_preds:
        if pred_lbl in cfg:
            for ins in reversed(cfg[pred_lbl].instrs):
                if isinstance(ins, Assign) and ins.dst == iv and isinstance(ins.src, int):
                    iv_init = ins.src
                    break

    # Separate loop body instructions from condition instructions
    cond_instr_ids = {id(branch_ins), id(cmp_ins)}
    raw_body_instrs = []
    for node in sorted(loop.body):
        for ins in cfg[node].instrs:
            if isinstance(ins, Label) or id(ins) in cond_instr_ids:
                continue
            if isinstance(ins, Goto) and ins.target in loop.body:
                continue
            raw_body_instrs.append(ins)

    # -----------------------------------------------------------------
    # Generate Vector Body Instructions
    # -----------------------------------------------------------------
    vector_body_instrs = []
    for ins in raw_body_instrs:
        if isinstance(ins, LoadIndexed):
            vector_body_instrs.append(VLoad(dst=ins.dst, array=ins.array, base_index=ins.index, width=4))
        elif isinstance(ins, StoreIndexed):
            vector_body_instrs.append(VStore(array=ins.array, base_index=ins.index, src=ins.src, width=4))
        elif isinstance(ins, BinOp):
            if ins.dst == iv and ins.op == "+":
                # Stride 4 induction increment
                vector_body_instrs.append(BinOp(dst=iv, op="+", a=iv, b=4))
            elif ins.op == "+":
                vector_body_instrs.append(VAdd(dst=ins.dst, a=ins.a, b=ins.b))
            elif ins.op == "-":
                vector_body_instrs.append(VSub(dst=ins.dst, a=ins.a, b=ins.b))
            elif ins.op == "*":
                vector_body_instrs.append(VMul(dst=ins.dst, a=ins.a, b=ins.b))
            elif ins.op == "/":
                vector_body_instrs.append(VDiv(dst=ins.dst, a=ins.a, b=ins.b))
            else:
                vector_body_instrs.append(ins)
        else:
            vector_body_instrs.append(ins)

    # -----------------------------------------------------------------
    # CFG Labels and Block Construction
    # -----------------------------------------------------------------
    setup_lbl = f"{header_lbl}_vsetup"
    vec_hdr_lbl = f"{header_lbl}_vhdr"
    vec_body_lbl = f"{header_lbl}_vbody"
    rem_hdr_lbl = f"{header_lbl}_rhdr"
    rem_body_lbl = f"{header_lbl}_rbody"

    v_rem_var = f"_v_rem_{header_lbl}"
    v_limit_var = f"_v_limit_{header_lbl}"
    v_cond_var = f"_v_cond_{header_lbl}"
    rem_cond_var = f"_r_cond_{header_lbl}"

    # Setup block: compute v_limit = limit - (limit % 4)
    setup_instrs = [
        Label(setup_lbl),
        BinOp(dst=v_rem_var, op="%", a=limit, b=4),
        BinOp(dst=v_limit_var, op="-", a=limit, b=v_rem_var),
        Goto(target=vec_hdr_lbl),
    ]

    # Vector loop header: branch to remainder header when iv >= v_limit
    vec_hdr_instrs = [
        Label(vec_hdr_lbl),
        BinOp(dst=v_cond_var, op=comp_op, a=iv, b=v_limit_var),
    ]
    if branch_exits_on_target:
        # Branch when condition is met -> exit to rem_hdr_lbl
        branch_cls = branch_ins.__class__
        vec_hdr_instrs.append(branch_cls(cond=v_cond_var, target=rem_hdr_lbl))
    else:
        # Fallthrough was exit -> jump to rem_hdr_lbl when condition is false (IfZ)
        vec_hdr_instrs.append(IfZ(cond=v_cond_var, target=rem_hdr_lbl))

    # Vector loop body: vector instructions, increment by 4, jump back to vec_hdr
    vec_body_instrs_full = [Label(vec_body_lbl)] + vector_body_instrs + [Goto(target=vec_hdr_lbl)]

    # Remainder loop header: branch to exit_target when iv >= limit
    rem_hdr_instrs = [
        Label(rem_hdr_lbl),
        BinOp(dst=rem_cond_var, op=comp_op, a=iv, b=limit),
    ]
    if branch_exits_on_target:
        branch_cls = branch_ins.__class__
        rem_hdr_instrs.append(branch_cls(cond=rem_cond_var, target=exit_target))
    else:
        rem_hdr_instrs.append(IfZ(cond=rem_cond_var, target=exit_target))

    # Remainder loop body: original scalar instructions, increment by 1, jump back to rem_hdr
    rem_body_instrs_full = [Label(rem_body_lbl)] + list(raw_body_instrs) + [Goto(target=rem_hdr_lbl)]

    # -----------------------------------------------------------------
    # Update CFG
    # -----------------------------------------------------------------
    # Create new blocks
    setup_block = BasicBlock(label=setup_lbl, instrs=setup_instrs, preds=list(outside_preds), succs=[vec_hdr_lbl])
    vec_hdr_block = BasicBlock(
        label=vec_hdr_lbl,
        instrs=vec_hdr_instrs,
        preds=[setup_lbl, vec_body_lbl],
        succs=[rem_hdr_lbl, vec_body_lbl],
    )
    vec_body_block = BasicBlock(
        label=vec_body_lbl,
        instrs=vec_body_instrs_full,
        preds=[vec_hdr_lbl],
        succs=[vec_hdr_lbl],
    )
    rem_hdr_block = BasicBlock(
        label=rem_hdr_lbl,
        instrs=rem_hdr_instrs,
        preds=[vec_hdr_lbl, rem_body_lbl],
        succs=[exit_target, rem_body_lbl],
    )
    rem_body_block = BasicBlock(
        label=rem_body_lbl,
        instrs=rem_body_instrs_full,
        preds=[rem_hdr_lbl],
        succs=[rem_hdr_lbl],
    )

    # Retarget outside predecessors to setup_block
    for pred_lbl in outside_preds:
        if pred_lbl in cfg:
            pred_b = cfg[pred_lbl]
            for ins in pred_b.instrs:
                if isinstance(ins, (Goto, IfZ, IfNz)) and ins.target == header_lbl:
                    ins.target = setup_lbl
            pred_b.succs = [setup_lbl if s == header_lbl else s for s in pred_b.succs]

    # Retarget exit block predecessor to rem_hdr_lbl
    if exit_target in cfg:
        exit_b = cfg[exit_target]
        exit_b.preds = [rem_hdr_lbl if p in loop.body else p for p in exit_b.preds]
        if rem_hdr_lbl not in exit_b.preds:
            exit_b.preds.append(rem_hdr_lbl)

    # Remove old loop blocks from CFG
    for node in loop.body:
        cfg.pop(node, None)

    # Insert new blocks into CFG
    cfg[setup_lbl] = setup_block
    cfg[vec_hdr_lbl] = vec_hdr_block
    cfg[vec_body_lbl] = vec_body_block
    cfg[rem_hdr_lbl] = rem_hdr_block
    cfg[rem_body_lbl] = rem_body_block

    return True


# =====================================================================
# STEP 4: Pipeline Drivers
# =====================================================================

def vectorize_function(
    fn: Function, stats: Optional[VectorizationStats] = None
) -> Tuple[Function, List[LoopVectorizationResult]]:
    """Analyze and vectorize loops in a single Function."""
    results: List[LoopVectorizationResult] = []
    cfg = build_cfg(fn)
    loops_found = find_natural_loops(cfg)

    # Sort innermost first
    sorted_loops = sorted(loops_found, key=lambda l: len(l.body))

    for loop in sorted_loops:
        if stats:
            stats.loops_analyzed += 1

        is_vec, reason, ctx = check_vectorizability(loop, cfg)
        if not is_vec:
            res = LoopVectorizationResult(
                header=loop.header,
                vectorized=False,
                reason=reason,
                speedup=1.0,
            )
            results.append(res)
            if stats:
                stats.details.append(res)
            continue

        # Calculate expected speedup
        limit = ctx["limit"]
        iv = ctx["iv"]
        trip_count = None
        if isinstance(limit, int):
            trip_count = max(0, limit)
            v_iters = trip_count // 4
            rem_iters = trip_count % 4
            total_iters = v_iters + rem_iters
            speedup = (trip_count / total_iters) if total_iters > 0 else 1.0
        else:
            speedup = 4.0  # Asymptotic SIMD width 4 speedup

        success = vectorize_loop(loop, cfg, fn.name, ctx)
        if success:
            res = LoopVectorizationResult(
                header=loop.header,
                vectorized=True,
                reason="vectorized with SIMD width 4 (remainder loop generated)",
                speedup=speedup,
                trip_count=trip_count,
            )
            results.append(res)
            if stats:
                stats.loops_vectorized += 1
                stats.details.append(res)

    new_fn = cfg_to_function(fn, cfg)
    return new_fn, results


def vectorize_program(
    functions: List[Function], stats: Optional[VectorizationStats] = None
) -> Tuple[List[Function], VectorizationStats]:
    """Run vectorizer across all functions in the program."""
    if stats is None:
        stats = VectorizationStats()

    new_functions = []
    for fn in functions:
        transformed_fn, _ = vectorize_function(fn, stats=stats)
        new_functions.append(transformed_fn)

    return new_functions, stats


# =====================================================================
# STEP 5: Dynamic Hooks for Codegen, Dataflow, and Regalloc
# =====================================================================

_orig_codegen_vars = codegen._vars
_orig_codegen_gen_function = codegen._gen_function
_orig_dataflow_read_vars = dataflow._get_read_vars
_orig_dataflow_def_var = dataflow._get_def_var
_orig_regalloc_replace_uses = getattr(regalloc, "_replace_uses", None)
_orig_regalloc_replace_def = getattr(regalloc, "_replace_def", None)


def _vector_vars(ins: Any) -> List[str]:
    if isinstance(ins, LoadIndexed):
        items = [ins.dst, ins.array, ins.index]
    elif isinstance(ins, StoreIndexed):
        items = [ins.array, ins.index, ins.src]
    elif isinstance(ins, NewArray):
        items = [ins.dst, ins.size]
    elif isinstance(ins, VLoad):
        items = [ins.dst, ins.array, ins.base_index]
    elif isinstance(ins, VStore):
        items = [ins.array, ins.base_index, ins.src]
    elif isinstance(ins, (VAdd, VSub, VMul, VDiv)):
        items = [ins.dst, ins.a, ins.b]
    else:
        return _orig_codegen_vars(ins)
    return [x for x in items if isinstance(x, str)]


def _ext_read_vars(ins: Any) -> List[str]:
    if isinstance(ins, LoadIndexed):
        return [v for v in (ins.array, ins.index) if isinstance(v, str)]
    elif isinstance(ins, StoreIndexed):
        return [v for v in (ins.array, ins.index, ins.src) if isinstance(v, str)]
    elif isinstance(ins, NewArray):
        return [ins.size] if isinstance(ins.size, str) else []
    elif isinstance(ins, VLoad):
        return [v for v in (ins.array, ins.base_index) if isinstance(v, str)]
    elif isinstance(ins, VStore):
        return [v for v in (ins.array, ins.base_index, ins.src) if isinstance(v, str)]
    elif isinstance(ins, (VAdd, VSub, VMul, VDiv)):
        return [v for v in (ins.a, ins.b) if isinstance(v, str)]
    return _orig_dataflow_read_vars(ins)


def _ext_def_var(ins: Any) -> Optional[str]:
    if isinstance(ins, (LoadIndexed, NewArray, VLoad, VAdd, VSub, VMul, VDiv)):
        return ins.dst
    if isinstance(ins, (StoreIndexed, VStore)):
        return None
    return _orig_dataflow_def_var(ins)


def _ext_replace_uses(ins: Any, rep_map: Dict[str, str]) -> Any:
    if isinstance(ins, LoadIndexed):
        return LoadIndexed(
            dst=ins.dst,
            array=rep_map.get(ins.array, ins.array) if isinstance(ins.array, str) else ins.array,
            index=rep_map.get(ins.index, ins.index) if isinstance(ins.index, str) else ins.index,
        )
    elif isinstance(ins, StoreIndexed):
        return StoreIndexed(
            array=rep_map.get(ins.array, ins.array) if isinstance(ins.array, str) else ins.array,
            index=rep_map.get(ins.index, ins.index) if isinstance(ins.index, str) else ins.index,
            src=rep_map.get(ins.src, ins.src) if isinstance(ins.src, str) else ins.src,
        )
    elif isinstance(ins, NewArray):
        return NewArray(
            dst=ins.dst,
            size=rep_map.get(ins.size, ins.size) if isinstance(ins.size, str) else ins.size,
        )
    elif isinstance(ins, VLoad):
        return VLoad(
            dst=ins.dst,
            array=rep_map.get(ins.array, ins.array) if isinstance(ins.array, str) else ins.array,
            base_index=rep_map.get(ins.base_index, ins.base_index) if isinstance(ins.base_index, str) else ins.base_index,
            width=ins.width,
        )
    elif isinstance(ins, VStore):
        return VStore(
            array=rep_map.get(ins.array, ins.array) if isinstance(ins.array, str) else ins.array,
            base_index=rep_map.get(ins.base_index, ins.base_index) if isinstance(ins.base_index, str) else ins.base_index,
            src=rep_map.get(ins.src, ins.src) if isinstance(ins.src, str) else ins.src,
            width=ins.width,
        )
    elif isinstance(ins, (VAdd, VSub, VMul, VDiv)):
        cls = ins.__class__
        return cls(
            dst=ins.dst,
            a=rep_map.get(ins.a, ins.a) if isinstance(ins.a, str) else ins.a,
            b=rep_map.get(ins.b, ins.b) if isinstance(ins.b, str) else ins.b,
        )
    if _orig_regalloc_replace_uses:
        return _orig_regalloc_replace_uses(ins, rep_map)
    return ins


def _ext_replace_def(ins: Any, new_dst: str) -> Any:
    if isinstance(ins, (LoadIndexed, NewArray, VLoad, VAdd, VSub, VMul, VDiv)):
        cls = ins.__class__
        kwargs = {k: v for k, v in ins.__dict__.items() if k not in ("dst", "line")}
        new_ins = cls(dst=new_dst, **kwargs)
        new_ins.line = getattr(ins, "line", 0)
        return new_ins
    if _orig_regalloc_replace_def:
        return _orig_regalloc_replace_def(ins, new_dst)
    return ins


def _vector_gen_function(fn, code, sigs, call_fixups, labels, reg_map=None):
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
        """Allocate or lookup stack slot index for variable name."""
        if name not in slots:
            next_slot = 8 if reg_map else 0
            while next_slot in slots.values():
                next_slot += 1
            slots[name] = next_slot
        return slots[name]

    for ins in fn.body:
        for v in codegen._vars(ins):
            slot(v)

    enter_at = len(code)
    code.append(Instr(Op.ENTER, 0))
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
        """Emit push instruction for constant or variable load."""
        if isinstance(x, int):
            if not -2**31 <= x < 2**31:
                raise codegen.CodegenError(f"constant {x} does not fit in 32 bits")
            code.append(Instr(Op.PUSH, x))
        else:
            code.append(Instr(Op.LOAD, slots[x]))

    def jump(op, target, line):
        """Emit jump instruction and record patch entry."""
        jump_fixups.append((len(code), target, line))
        code.append(Instr(op, 0))

    for ins in fn.body:
        if isinstance(ins, Label):
            if ins.name in local_labels:
                raise codegen.CodegenError(f"line {ins.line}: duplicate label '{ins.name}'")
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
            code.append(Instr(codegen.BINOP_CODE[ins.op]))
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
                raise codegen.CodegenError(f"line {ins.line}: call to unknown function '{ins.func}'")
            if len(ins.args) != sigs[ins.func]:
                raise codegen.CodegenError(
                    f"line {ins.line}: '{ins.func}' expects {sigs[ins.func]} argument(s), "
                    f"got {len(ins.args)}"
                )
            for a in ins.args:
                push(a)
            call_fixups.append((len(code), ins.func))
            code.append(Instr(Op.CALL, 0, len(ins.args)))
            if ins.dst:
                code.append(Instr(Op.STORE, slot(ins.dst)))
            else:
                code.append(Instr(Op.POP))
        # Vector and Array Operations
        elif isinstance(ins, LoadIndexed):
            push(ins.index)
            code.append(Instr(Op.LOAD_INDEXED, slot(ins.array)))
            code.append(Instr(Op.STORE, slot(ins.dst)))
        elif isinstance(ins, StoreIndexed):
            push(ins.index)
            push(ins.src)
            code.append(Instr(Op.STORE_INDEXED, slot(ins.array)))
        elif isinstance(ins, NewArray):
            push(ins.size)
            code.append(Instr(Op.NEW_ARRAY))
            code.append(Instr(Op.STORE, slot(ins.dst)))
        elif isinstance(ins, VLoad):
            push(ins.base_index)
            code.append(Instr(Op.VLOAD, slot(ins.array), ins.width))
            code.append(Instr(Op.STORE, slot(ins.dst)))
        elif isinstance(ins, VStore):
            push(ins.base_index)
            push(ins.src)
            code.append(Instr(Op.VSTORE, slot(ins.array), ins.width))
        elif isinstance(ins, VAdd):
            push(ins.a)
            push(ins.b)
            code.append(Instr(Op.VADD))
            code.append(Instr(Op.STORE, slot(ins.dst)))
        elif isinstance(ins, VSub):
            push(ins.a)
            push(ins.b)
            code.append(Instr(Op.VSUB))
            code.append(Instr(Op.STORE, slot(ins.dst)))
        elif isinstance(ins, VMul):
            push(ins.a)
            push(ins.b)
            code.append(Instr(Op.VMUL))
            code.append(Instr(Op.STORE, slot(ins.dst)))
        elif isinstance(ins, VDiv):
            push(ins.a)
            push(ins.b)
            code.append(Instr(Op.VDIV))
            code.append(Instr(Op.STORE, slot(ins.dst)))
        else:
            raise codegen.CodegenError(f"unsupported IR node {ins!r}")

    if not fn.body or not isinstance(fn.body[-1], Return):
        code.append(Instr(Op.PUSH, 0))
        code.append(Instr(Op.RET))

    for idx, name, line in jump_fixups:
        if name not in local_labels:
            raise codegen.CodegenError(f"line {line}: undefined label '{name}' in function '{fn.name}'")
        code[idx] = Instr(code[idx].op, local_labels[name])

    final_slots_count = max(8 if reg_map else 0, max(slots.values()) + 1 if slots else 0)
    code[enter_at] = Instr(Op.ENTER, final_slots_count)
    for name, addr in local_labels.items():
        labels.setdefault(addr, []).append(name)


def _install_dynamic_hooks():
    codegen._vars = _vector_vars
    codegen._gen_function = _vector_gen_function
    dataflow._get_read_vars = _ext_read_vars
    dataflow._get_def_var = _ext_def_var
    import sys
    opt = sys.modules.get("optimizer")
    if opt:
        if hasattr(opt, "_get_read_vars"):
            opt._get_read_vars = _ext_read_vars
        if hasattr(opt, "_get_def_var"):
            opt._get_def_var = _ext_def_var

    if hasattr(loops, "_get_def_var"):
        loops._get_def_var = _ext_def_var
    if hasattr(regalloc, "_get_def_var"):
        regalloc._get_def_var = _ext_def_var
    if hasattr(regalloc, "_get_read_vars"):
        regalloc._get_read_vars = _ext_read_vars
    if hasattr(regalloc, "_replace_uses"):
        regalloc._replace_uses = _ext_replace_uses
    if hasattr(regalloc, "_replace_def"):
        regalloc._replace_def = _ext_replace_def
    if hasattr(scheduler, "_get_read_vars"):
        scheduler._get_read_vars = _ext_read_vars
    if hasattr(scheduler, "_get_def_var"):
        scheduler._get_def_var = _ext_def_var


_install_dynamic_hooks()
