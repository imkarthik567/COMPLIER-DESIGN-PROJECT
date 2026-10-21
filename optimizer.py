"""CVM Optimizing Compiler Backend.

Provides:
- OptimizationPass: Base class with standard pass interface
- Pass 1: DeadCodeElimination (uses LivenessAnalysis)
- Pass 2: ConstantFoldingAndPropagation (uses ReachingDefinitions)
- Pass 3: CopyPropagation (uses ReachingDefinitions)
- Pass 4: CommonSubexpressionElimination (uses AvailableExpressions)
- Pass 5: LoopInvariantCodeMotion (uses NaturalLoop & Dominators)
- Pass 6: StrengthReduction (uses Induction Variables)
- Pass 7: LoopUnrolling (uses Loop Detection & Trip Counts)
- OptimizationPipeline: Pipeline orchestrator with fixed-point convergence
- BasicBlock, build_cfg, cfg_to_function: CFG utilities
"""
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ir import (
    Assign, BinOp, Call, Function, Goto, IfNz, IfZ, Label, Print, Return,
    LoadIndexed, StoreIndexed, NewArray, VLoad, VStore, VAdd, VSub, VMul, VDiv
)

Operand = Union[int, str]


# =====================================================================
# CFG Data Structures and Utilities
# =====================================================================

@dataclass
class BasicBlock:
    """Represents a basic block in the control-flow graph."""
    label: str
    instrs: list = field(default_factory=list)
    preds: list = field(default_factory=list)
    succs: list = field(default_factory=list)


@dataclass
class OptimizationStats:
    """Aggregated metrics across optimization passes."""
    constants_folded: int = 0
    algebraic_simplified: int = 0
    copies_propagated: int = 0
    expressions_eliminated: int = 0
    instructions_hoisted: int = 0
    operations_reduced: int = 0
    loops_unrolled: int = 0
    branches_folded: int = 0
    dead_instrs_removed: int = 0
    unreachable_blocks_removed: int = 0

    def total(self) -> int:
        """Return the total number of optimizations applied across all categories."""
        return (
            self.constants_folded
            + self.algebraic_simplified
            + self.copies_propagated
            + self.expressions_eliminated
            + self.instructions_hoisted
            + self.operations_reduced
            + self.loops_unrolled
            + self.branches_folded
            + self.dead_instrs_removed
            + self.unreachable_blocks_removed
        )


from dataflow import (
    AvailableExpressions,
    DataFlowAnalysis,
    DataFlowResults,
    LivenessAnalysis,
    ReachingDefinitions,
    _get_def_var,
    _get_read_vars,
)
from loops import (
    NaturalLoop,
    compute_dominators,
    dominates,
    find_natural_loops,
    find_primary_induction_var,
)


def build_cfg(fn: Function) -> Dict[str, BasicBlock]:
    """Construct CFG dictionary mapping { block_label: BasicBlock } for a Function."""
    body = fn.body
    if not body:
        return {}

    leaders = {0}
    for i, ins in enumerate(body):
        if isinstance(ins, Label):
            leaders.add(i)
        if isinstance(ins, (Goto, IfZ, IfNz, Return)) and i + 1 < len(body):
            leaders.add(i + 1)
    starts = sorted(leaders)

    raw_blocks = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(body)
        raw_blocks.append((k, body[s:e]))

    label_to_block = {}
    blocks = []
    for k, chunk in raw_blocks:
        first = chunk[0] if chunk else None
        if isinstance(first, Label):
            blabel = first.name
        else:
            blabel = f"{fn.name}_entry" if k == 0 else f"{fn.name}_b{k}"

        for ins in chunk:
            if isinstance(ins, Label):
                label_to_block[ins.name] = blabel
        blocks.append((blabel, chunk))

    cfg: Dict[str, BasicBlock] = {}
    for blabel, chunk in blocks:
        cfg[blabel] = BasicBlock(label=blabel, instrs=list(chunk))

    for idx, (blabel, chunk) in enumerate(blocks):
        b = cfg[blabel]
        last = chunk[-1] if chunk else None
        nxt = [blocks[idx + 1][0]] if idx + 1 < len(blocks) else []

        if isinstance(last, Goto):
            b.succs = [label_to_block.get(last.target, last.target)]
        elif isinstance(last, (IfZ, IfNz)):
            target = label_to_block.get(last.target, last.target)
            b.succs = [target] + nxt
        elif isinstance(last, Return):
            b.succs = []
        else:
            b.succs = nxt

    for b in cfg.values():
        for s in b.succs:
            if s in cfg and b.label not in cfg[s].preds:
                cfg[s].preds.append(b.label)

    return cfg


def cfg_to_function(fn: Function, cfg: Dict[str, BasicBlock], block_order: Optional[List[str]] = None) -> Function:
    """Rebuild a Function instruction stream from CFG basic blocks."""
    if block_order is not None:
        order = block_order
    else:
        order = []
        for lbl in cfg.keys():
            if lbl.endswith("_preheader"):
                continue
            pre = f"{lbl}_preheader"
            if pre in cfg and pre not in order:
                order.append(pre)
            order.append(lbl)
        for lbl in cfg.keys():
            if lbl not in order:
                order.append(lbl)

    active_blocks = [cfg[lbl] for lbl in order if lbl in cfg]

    jump_targets = set()
    for b in active_blocks:
        for ins in b.instrs:
            if isinstance(ins, (Goto, IfZ, IfNz)):
                jump_targets.add(ins.target)

    for idx, b in enumerate(active_blocks):
        if not b.instrs:
            continue
        last = b.instrs[-1]
        if not isinstance(last, (Goto, Return)):
            expected_next = active_blocks[idx + 1].label if idx + 1 < len(active_blocks) else None
            actual_succ = b.succs[-1] if b.succs else None
            if actual_succ and actual_succ != expected_next:
                jump_targets.add(actual_succ)

    new_body = []
    for idx, b in enumerate(active_blocks):
        if not b.instrs:
            continue
        first = b.instrs[0]
        if b.label in jump_targets and not (isinstance(first, Label) and first.name == b.label):
            new_body.append(Label(b.label))

        new_body.extend(b.instrs)

        last = b.instrs[-1]
        if not isinstance(last, (Goto, Return)):
            expected_next = active_blocks[idx + 1].label if idx + 1 < len(active_blocks) else None
            actual_succ = b.succs[-1] if b.succs else None
            if actual_succ and actual_succ != expected_next:
                new_body.append(Goto(actual_succ))

    return Function(name=fn.name, params=fn.params, body=new_body, line=fn.line)


def eval_div(a: int, b: int) -> int:
    """Truncating division matching CVM VM semantics."""
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def eval_mod(a: int, b: int) -> int:
    """Remainder of truncating division."""
    return a - b * eval_div(a, b)


def fold_binop(op: str, a: int, b: int) -> Optional[int]:
    """Fold binary operations if safe and within 32-bit signed range."""
    if op == "+":
        res = a + b
    elif op == "-":
        res = a - b
    elif op == "*":
        res = a * b
    elif op == "/":
        if b == 0:
            return None  # Preserve runtime trap
        res = eval_div(a, b)
    elif op == "%":
        if b == 0:
            return None  # Preserve runtime trap
        res = eval_mod(a, b)
    elif op == "<":
        res = int(a < b)
    elif op == "<=":
        res = int(a <= b)
    elif op == ">":
        res = int(a > b)
    elif op == ">=":
        res = int(a >= b)
    elif op == "==":
        res = int(a == b)
    elif op == "!=":
        res = int(a != b)
    else:
        return None

    if -2**31 <= res < 2**31:
        return res
    return None


# =====================================================================
# PASS INTERFACE & IMPLEMENTATIONS
# =====================================================================

class OptimizationPass:
    """Standard optimization pass interface."""
    name: str = "BasePass"
    description: str = "Base optimization pass"
    required_analysis: Optional[type] = None

    def run(
        self,
        cfg: Dict[str, BasicBlock],
        dataflow_results: Optional[DataFlowResults] = None,
    ) -> Tuple[Dict[str, BasicBlock], Dict[str, int]]:
        """Run the optimization pass and return (new_cfg, stats_dict)."""
        raise NotImplementedError


class DeadCodeElimination(OptimizationPass):
    """Pass 1: Removes assignments to variables that are not live."""
    name = "Dead Code Elimination"
    description = "Removes dead assignments using backward liveness analysis"
    required_analysis = LivenessAnalysis

    def run(
        self,
        cfg: Dict[str, BasicBlock],
        dataflow_results: Optional[DataFlowResults] = None,
    ) -> Tuple[Dict[str, BasicBlock], Dict[str, int]]:
        """Execute dead code elimination pass across the CFG."""
        if dataflow_results is None and self.required_analysis is not None:
            dataflow_results = self.required_analysis().run(cfg)

        stats = {"instructions_removed": 0, "instructions_changed": 0}
        new_cfg: Dict[str, BasicBlock] = {}

        for lbl, b in cfg.items():
            live_out = set(dataflow_results.out_sets.get(lbl, set())) if dataflow_results else set()
            live_now = set(live_out)
            new_instrs = []

            for ins in reversed(b.instrs):
                if isinstance(ins, (Assign, BinOp)):
                    dst = ins.dst
                    if dst not in live_now:
                        stats["instructions_removed"] += 1
                        continue
                    live_now.discard(dst)
                    for v in _get_read_vars(ins):
                        live_now.add(v)
                    new_instrs.append(ins)
                elif isinstance(ins, Call):
                    dst = ins.dst
                    if dst:
                        live_now.discard(dst)
                    for a in ins.args:
                        if isinstance(a, str):
                            live_now.add(a)
                    new_instrs.append(ins)
                else:
                    for v in _get_read_vars(ins):
                        live_now.add(v)
                    new_instrs.append(ins)

            new_instrs.reverse()
            new_cfg[lbl] = BasicBlock(
                label=lbl,
                instrs=new_instrs,
                preds=list(b.preds),
                succs=list(b.succs),
            )

        return new_cfg, stats


class ConstantFoldingAndPropagation(OptimizationPass):
    """Pass 2: Evaluates constant expressions and propagates constant values."""
    name = "Constant Folding & Propagation"
    description = "Folds constant expressions and propagates reaching constant values"
    required_analysis = ReachingDefinitions

    def run(
        self,
        cfg: Dict[str, BasicBlock],
        dataflow_results: Optional[DataFlowResults] = None,
    ) -> Tuple[Dict[str, BasicBlock], Dict[str, int]]:
        """Execute constant folding and constant propagation pass across the CFG."""
        if dataflow_results is None and self.required_analysis is not None:
            dataflow_results = self.required_analysis().run(cfg)

        stats = {
            "instructions_removed": 0,
            "instructions_changed": 0,
            "constants_folded": 0,
            "constants_propagated": 0,
        }
        new_cfg: Dict[str, BasicBlock] = {}

        for lbl, b in cfg.items():
            local_consts: Dict[str, Optional[int]] = {}
            new_instrs = []

            for ins in b.instrs:
                def resolve_const(opnd: Operand) -> Operand:
                    """Resolve operand to known constant value if reaching definitions permit."""
                    if isinstance(opnd, int):
                        return opnd
                    if opnd in local_consts:
                        val = local_consts[opnd]
                        return val if val is not None else opnd

                    if dataflow_results is not None:
                        reaching = [d for d in dataflow_results.in_sets.get(lbl, set()) if d[0] == opnd]
                        if reaching:
                            const_vals = set()
                            for var, d_lbl, d_idx in reaching:
                                if d_lbl in cfg and d_idx < len(cfg[d_lbl].instrs):
                                    d_ins = cfg[d_lbl].instrs[d_idx]
                                    if isinstance(d_ins, Assign) and isinstance(d_ins.src, int):
                                        const_vals.add(d_ins.src)
                                    elif isinstance(d_ins, BinOp) and isinstance(d_ins.a, int) and isinstance(d_ins.b, int):
                                        folded = fold_binop(d_ins.op, d_ins.a, d_ins.b)
                                        if folded is not None:
                                            const_vals.add(folded)
                                        else:
                                            return opnd
                                    else:
                                        return opnd
                                else:
                                    return opnd
                            if len(const_vals) == 1:
                                return next(iter(const_vals))
                    return opnd

                changed_this_ins = False

                if isinstance(ins, Assign):
                    new_src = resolve_const(ins.src)
                    if new_src != ins.src and isinstance(new_src, int):
                        ins = Assign(ins.dst, new_src)
                        stats["constants_propagated"] += 1
                        changed_this_ins = True
                elif isinstance(ins, BinOp):
                    new_a = resolve_const(ins.a)
                    new_b = resolve_const(ins.b)
                    if (new_a != ins.a and isinstance(new_a, int)) or (new_b != ins.b and isinstance(new_b, int)):
                        ins = BinOp(ins.dst, ins.op, new_a, new_b)
                        stats["constants_propagated"] += 1
                        changed_this_ins = True
                elif isinstance(ins, (IfZ, IfNz)):
                    new_cond = resolve_const(ins.cond)
                    if new_cond != ins.cond and isinstance(new_cond, int):
                        cls = IfZ if isinstance(ins, IfZ) else IfNz
                        ins = cls(new_cond, ins.target)
                        stats["constants_propagated"] += 1
                        changed_this_ins = True
                elif isinstance(ins, Print):
                    new_arg = resolve_const(ins.arg)
                    if new_arg != ins.arg and isinstance(new_arg, int):
                        ins = Print(new_arg)
                        stats["constants_propagated"] += 1
                        changed_this_ins = True
                elif isinstance(ins, Return):
                    if ins.arg is not None:
                        new_arg = resolve_const(ins.arg)
                        if new_arg != ins.arg and isinstance(new_arg, int):
                            ins = Return(new_arg)
                            stats["constants_propagated"] += 1
                            changed_this_ins = True
                elif isinstance(ins, Call):
                    new_args = [resolve_const(a) for a in ins.args]
                    if new_args != ins.args:
                        ins = Call(ins.dst, ins.func, new_args)
                        stats["constants_propagated"] += 1
                        changed_this_ins = True
                elif isinstance(ins, LoadIndexed):
                    new_idx = resolve_const(ins.index)
                    if new_idx != ins.index and isinstance(new_idx, int):
                        ins = LoadIndexed(ins.dst, ins.array, new_idx)
                        stats["constants_propagated"] += 1
                        changed_this_ins = True
                elif isinstance(ins, StoreIndexed):
                    new_idx = resolve_const(ins.index)
                    new_src = resolve_const(ins.src)
                    if (new_idx != ins.index and isinstance(new_idx, int)) or (new_src != ins.src and isinstance(new_src, int)):
                        ins = StoreIndexed(ins.array, new_idx, new_src)
                        stats["constants_propagated"] += 1
                        changed_this_ins = True
                elif isinstance(ins, NewArray):
                    new_size = resolve_const(ins.size)
                    if new_size != ins.size and isinstance(new_size, int):
                        ins = NewArray(ins.dst, new_size)
                        stats["constants_propagated"] += 1
                        changed_this_ins = True

                if isinstance(ins, BinOp):
                    if isinstance(ins.a, int) and isinstance(ins.b, int):
                        folded = fold_binop(ins.op, ins.a, ins.b)
                        if folded is not None:
                            ins = Assign(ins.dst, folded)
                            stats["constants_folded"] += 1
                            changed_this_ins = True

                if changed_this_ins:
                    stats["instructions_changed"] += 1

                dst = _get_def_var(ins)
                if dst:
                    if isinstance(ins, Assign) and isinstance(ins.src, int):
                        local_consts[dst] = ins.src
                    else:
                        local_consts[dst] = None

                new_instrs.append(ins)

            new_cfg[lbl] = BasicBlock(
                label=lbl,
                instrs=new_instrs,
                preds=list(b.preds),
                succs=list(b.succs),
            )

        return new_cfg, stats


class CopyPropagation(OptimizationPass):
    """Pass 3: Propagates variable copies (x = y) to later uses."""
    name = "Copy Propagation"
    description = "Replaces variable uses with their original source copy"
    required_analysis = ReachingDefinitions

    def run(
        self,
        cfg: Dict[str, BasicBlock],
        dataflow_results: Optional[DataFlowResults] = None,
    ) -> Tuple[Dict[str, BasicBlock], Dict[str, int]]:
        """Execute copy propagation pass across the CFG."""
        if dataflow_results is None and self.required_analysis is not None:
            dataflow_results = self.required_analysis().run(cfg)

        stats = {
            "instructions_removed": 0,
            "instructions_changed": 0,
            "copies_propagated": 0,
        }
        new_cfg: Dict[str, BasicBlock] = {}

        for lbl, b in cfg.items():
            local_copies: Dict[str, Optional[str]] = {}
            new_instrs = []

            for ins in b.instrs:
                def resolve_copy(opnd: Operand) -> Operand:
                    """Resolve operand through copy chains to original source variable."""
                    if not isinstance(opnd, str):
                        return opnd

                    curr = opnd
                    visited = {curr}
                    while curr in local_copies:
                        nxt = local_copies[curr]
                        if nxt is None or nxt in visited:
                            break
                        curr = nxt
                        visited.add(curr)
                    if curr != opnd:
                        return curr

                    if dataflow_results is not None:
                        reaching = [d for d in dataflow_results.in_sets.get(lbl, set()) if d[0] == opnd]
                        if len(reaching) == 1:
                            var, d_lbl, d_idx = reaching[0]
                            if d_lbl in cfg and d_idx < len(cfg[d_lbl].instrs):
                                d_ins = cfg[d_lbl].instrs[d_idx]
                                if isinstance(d_ins, Assign) and isinstance(d_ins.src, str):
                                    src_var = d_ins.src
                                    if local_copies.get(src_var, "ok") is not None:
                                        src_reaching = [d for d in dataflow_results.in_sets.get(lbl, set()) if d[0] == src_var]
                                        if len(src_reaching) <= 1:
                                            return src_var
                    return opnd

                changed_this_ins = False

                if isinstance(ins, Assign):
                    new_src = resolve_copy(ins.src)
                    if new_src != ins.src and isinstance(new_src, str):
                        ins = Assign(ins.dst, new_src)
                        stats["copies_propagated"] += 1
                        changed_this_ins = True
                elif isinstance(ins, BinOp):
                    new_a = resolve_copy(ins.a)
                    new_b = resolve_copy(ins.b)
                    if (new_a != ins.a and isinstance(new_a, str)) or (new_b != ins.b and isinstance(new_b, str)):
                        ins = BinOp(ins.dst, ins.op, new_a, new_b)
                        stats["copies_propagated"] += 1
                        changed_this_ins = True
                elif isinstance(ins, (IfZ, IfNz)):
                    new_cond = resolve_copy(ins.cond)
                    if new_cond != ins.cond and isinstance(new_cond, str):
                        cls = IfZ if isinstance(ins, IfZ) else IfNz
                        ins = cls(new_cond, ins.target)
                        stats["copies_propagated"] += 1
                        changed_this_ins = True
                elif isinstance(ins, Print):
                    new_arg = resolve_copy(ins.arg)
                    if new_arg != ins.arg and isinstance(new_arg, str):
                        ins = Print(new_arg)
                        stats["copies_propagated"] += 1
                        changed_this_ins = True
                elif isinstance(ins, Return):
                    if ins.arg is not None:
                        new_arg = resolve_copy(ins.arg)
                        if new_arg != ins.arg and isinstance(new_arg, str):
                            ins = Return(new_arg)
                            stats["copies_propagated"] += 1
                            changed_this_ins = True
                elif isinstance(ins, Call):
                    new_args = [resolve_copy(a) for a in ins.args]
                    if new_args != ins.args:
                        ins = Call(ins.dst, ins.func, new_args)
                        stats["copies_propagated"] += 1
                        changed_this_ins = True
                elif isinstance(ins, LoadIndexed):
                    new_idx = resolve_copy(ins.index)
                    new_arr = resolve_copy(ins.array)
                    if (new_idx != ins.index) or (new_arr != ins.array and isinstance(new_arr, str)):
                        ins = LoadIndexed(ins.dst, new_arr if isinstance(new_arr, str) else ins.array, new_idx)
                        stats["copies_propagated"] += 1
                        changed_this_ins = True
                elif isinstance(ins, StoreIndexed):
                    new_idx = resolve_copy(ins.index)
                    new_src = resolve_copy(ins.src)
                    new_arr = resolve_copy(ins.array)
                    if (new_idx != ins.index) or (new_src != ins.src) or (new_arr != ins.array):
                        ins = StoreIndexed(new_arr if isinstance(new_arr, str) else ins.array, new_idx, new_src)
                        stats["copies_propagated"] += 1
                        changed_this_ins = True
                elif isinstance(ins, NewArray):
                    new_size = resolve_copy(ins.size)
                    if new_size != ins.size:
                        ins = NewArray(ins.dst, new_size)
                        stats["copies_propagated"] += 1
                        changed_this_ins = True

                if changed_this_ins:
                    stats["instructions_changed"] += 1

                dst = _get_def_var(ins)
                if dst:
                    local_copies.pop(dst, None)
                    for k in [k for k, v in local_copies.items() if v == dst]:
                        local_copies[k] = None

                if isinstance(ins, Assign) and isinstance(ins.src, str) and ins.src != ins.dst:
                    local_copies[ins.dst] = ins.src

                new_instrs.append(ins)

            new_cfg[lbl] = BasicBlock(
                label=lbl,
                instrs=new_instrs,
                preds=list(b.preds),
                succs=list(b.succs),
            )

        return new_cfg, stats


class CommonSubexpressionElimination(OptimizationPass):
    """Pass 4: Replaces recomputed expressions with copies of previously computed temporaries."""
    name = "Common Subexpression Elimination"
    description = "Eliminates redundant binary expressions using Available Expressions"
    required_analysis = AvailableExpressions

    def run(
        self,
        cfg: Dict[str, BasicBlock],
        dataflow_results: Optional[DataFlowResults] = None,
    ) -> Tuple[Dict[str, BasicBlock], Dict[str, int]]:
        """Execute common subexpression elimination pass across the CFG."""
        if dataflow_results is None and self.required_analysis is not None:
            dataflow_results = self.required_analysis().run(cfg)

        stats = {
            "instructions_removed": 0,
            "instructions_changed": 0,
            "expressions_eliminated": 0,
        }
        new_cfg: Dict[str, BasicBlock] = {}

        for lbl, b in cfg.items():
            avail_at_entry = set(dataflow_results.in_sets.get(lbl, set())) if dataflow_results else set()
            local_expr_to_var: Dict[Tuple[str, Any, Any], str] = {}
            modified_vars: Set[str] = set()
            new_instrs = []

            for ins in b.instrs:
                changed = False
                if isinstance(ins, BinOp):
                    expr = (ins.op, ins.a, ins.b)
                    if expr in local_expr_to_var:
                        prev_var = local_expr_to_var[expr]
                        ins = Assign(ins.dst, prev_var)
                        stats["expressions_eliminated"] += 1
                        stats["instructions_changed"] += 1
                        changed = True
                    elif expr in avail_at_entry and ins.a not in modified_vars and ins.b not in modified_vars:
                        candidates = []
                        for p_lbl in b.preds:
                            if p_lbl in cfg:
                                for p_ins in reversed(cfg[p_lbl].instrs):
                                    if isinstance(p_ins, BinOp) and (p_ins.op, p_ins.a, p_ins.b) == expr:
                                        candidates.append(p_ins.dst)
                                        break
                        if candidates and all(c == candidates[0] for c in candidates):
                            prev_var = candidates[0]
                            ins = Assign(ins.dst, prev_var)
                            stats["expressions_eliminated"] += 1
                            stats["instructions_changed"] += 1
                            changed = True

                    if not changed:
                        local_expr_to_var[expr] = ins.dst

                dst = _get_def_var(ins)
                if dst:
                    modified_vars.add(dst)
                    dead_exprs = [e for e in local_expr_to_var if e[1] == dst or e[2] == dst]
                    for e in dead_exprs:
                        del local_expr_to_var[e]

                new_instrs.append(ins)

            new_cfg[lbl] = BasicBlock(
                label=lbl,
                instrs=new_instrs,
                preds=list(b.preds),
                succs=list(b.succs),
            )

        return new_cfg, stats


class LoopInvariantCodeMotion(OptimizationPass):
    """Pass 5: Hoists loop-invariant instructions to a preheader block."""
    name = "Loop-Invariant Code Motion"
    description = "Hoists computations whose operands are loop-invariant to a preheader"
    required_analysis = ReachingDefinitions

    def run(
        self,
        cfg: Dict[str, BasicBlock],
        dataflow_results: Optional[DataFlowResults] = None,
    ) -> Tuple[Dict[str, BasicBlock], Dict[str, int]]:
        """Execute loop-invariant code motion pass across the CFG."""
        if dataflow_results is None and self.required_analysis is not None:
            dataflow_results = self.required_analysis().run(cfg)

        stats = {
            "instructions_removed": 0,
            "instructions_changed": 0,
            "instructions_hoisted": 0,
        }
        loops = find_natural_loops(cfg)
        if not loops:
            return cfg, stats

        idom, dom, _ = compute_dominators(cfg)
        # Deep copy blocks so we don't mutate in place if an issue arises
        new_cfg: Dict[str, BasicBlock] = {
            lbl: BasicBlock(label=lbl, instrs=list(b.instrs), preds=list(b.preds), succs=list(b.succs))
            for lbl, b in cfg.items()
        }

        # Process loops innermost to outermost
        sorted_loops = sorted(loops, key=lambda l: len(l.body))

        for loop in sorted_loops:
            header = loop.header
            if header not in new_cfg:
                continue

            # 1. Identify hoistable loop-invariant instructions
            hoistable: List[Tuple[str, int, Any]] = []
            hoisted_vars: Set[str] = set()

            # Count variable definitions inside loop
            loop_defs: Dict[str, int] = {}
            for node in loop.body:
                if node in new_cfg:
                    for ins in new_cfg[node].instrs:
                        dst = _get_def_var(ins)
                        if dst:
                            loop_defs[dst] = loop_defs.get(dst, 0) + 1

            changed = True
            while changed:
                changed = False
                for node in loop.body:
                    if node not in new_cfg:
                        continue
                    # Block must dominate all loop exit blocks
                    if not all(dominates(node, exit_b, dom) for exit_b in loop.exit_blocks if exit_b in new_cfg):
                        continue

                    for idx, ins in enumerate(new_cfg[node].instrs):
                        if not isinstance(ins, (Assign, BinOp)):
                            continue
                        dst = ins.dst
                        if dst in hoisted_vars:
                            continue
                        if loop_defs.get(dst, 0) != 1:
                            continue  # Must be defined exactly once in the loop

                        # Check if all operands are loop-invariant
                        reads = _get_read_vars(ins)
                        invariant = True
                        for r in reads:
                            if r in hoisted_vars:
                                continue
                            if loop_defs.get(r, 0) > 0:
                                invariant = False
                                break
                            if dataflow_results is not None:
                                reaching = [d for d in dataflow_results.in_sets.get(node, set()) if d[0] == r]
                                if any(d[1] in loop.body for d in reaching):
                                    invariant = False
                                    break

                        if invariant:
                            hoistable.append((node, idx, ins))
                            hoisted_vars.add(dst)
                            changed = True

            if not hoistable:
                continue

            # 2. Insert or find preheader block
            preheader_label = f"{header}_preheader"
            if preheader_label not in new_cfg:
                preds_out = [p for p in new_cfg[header].preds if p not in loop.body]
                preds_in = [p for p in new_cfg[header].preds if p in loop.body]
                preheader = BasicBlock(
                    label=preheader_label,
                    instrs=[],
                    preds=list(preds_out),
                    succs=[header],
                )
                new_cfg[preheader_label] = preheader

                # Redirect outside predecessors to preheader
                for p in preds_out:
                    if p in new_cfg:
                        new_cfg[p].succs = [preheader_label if s == header else s for s in new_cfg[p].succs]
                        for p_ins in new_cfg[p].instrs:
                            if isinstance(p_ins, Goto) and p_ins.target == header:
                                p_ins.target = preheader_label
                            elif isinstance(p_ins, (IfZ, IfNz)) and p_ins.target == header:
                                p_ins.target = preheader_label

                new_cfg[header].preds = [preheader_label] + preds_in

            # 3. Move instructions into preheader
            hoisted_instrs = [item[2] for item in hoistable]
            new_cfg[preheader_label].instrs.extend(hoisted_instrs)

            # Remove hoisted instructions from original blocks
            hoist_set = set(id(ins) for ins in hoisted_instrs)
            for node in loop.body:
                if node in new_cfg:
                    new_cfg[node].instrs = [ins for ins in new_cfg[node].instrs if id(ins) not in hoist_set]

            stats["instructions_hoisted"] += len(hoisted_instrs)
            stats["instructions_changed"] += len(hoisted_instrs)

        return new_cfg, stats


class StrengthReduction(OptimizationPass):
    """Pass 6: Replaces expensive loop operations (multiplication by induction variable) with repeated additions."""
    name = "Strength Reduction"
    description = "Replaces induction variable multiplication with incremental additions"
    required_analysis = None

    def run(
        self,
        cfg: Dict[str, BasicBlock],
        dataflow_results: Optional[DataFlowResults] = None,
    ) -> Tuple[Dict[str, BasicBlock], Dict[str, int]]:
        """Execute induction variable strength reduction pass across the CFG."""
        stats = {
            "instructions_removed": 0,
            "instructions_changed": 0,
            "operations_reduced": 0,
        }
        loops = find_natural_loops(cfg)
        if not loops:
            return cfg, stats

        new_cfg: Dict[str, BasicBlock] = {
            lbl: BasicBlock(label=lbl, instrs=list(b.instrs), preds=list(b.preds), succs=list(b.succs))
            for lbl, b in cfg.items()
        }

        for loop in loops:
            iv_info = find_primary_induction_var(loop, new_cfg)
            if not iv_info:
                continue

            iv_var, step, inc_block, inc_idx, inc_ins = iv_info

            # Find candidate multiplication t = i * k or t = k * i
            candidates = []
            for node in loop.body:
                if node in new_cfg:
                    for idx, ins in enumerate(new_cfg[node].instrs):
                        if isinstance(ins, BinOp) and ins.op == "*":
                            if ins.a == iv_var and isinstance(ins.b, int):
                                candidates.append((node, idx, ins, ins.b))
                            elif ins.b == iv_var and isinstance(ins.a, int):
                                candidates.append((node, idx, ins, ins.a))

            if not candidates:
                continue

            header = loop.header
            preheader_label = f"{header}_preheader"
            if preheader_label not in new_cfg:
                preds_out = [p for p in new_cfg[header].preds if p not in loop.body]
                preds_in = [p for p in new_cfg[header].preds if p in loop.body]
                preheader = BasicBlock(
                    label=preheader_label,
                    instrs=[],
                    preds=list(preds_out),
                    succs=[header],
                )
                new_cfg[preheader_label] = preheader

                for p in preds_out:
                    if p in new_cfg:
                        new_cfg[p].succs = [preheader_label if s == header else s for s in new_cfg[p].succs]
                        for p_ins in new_cfg[p].instrs:
                            if isinstance(p_ins, Goto) and p_ins.target == header:
                                p_ins.target = preheader_label
                            elif isinstance(p_ins, (IfZ, IfNz)) and p_ins.target == header:
                                p_ins.target = preheader_label

                new_cfg[header].preds = [preheader_label] + preds_in

            for node, idx, ins, k in candidates:
                red_var = f"{ins.dst}_red"
                delta = step * k

                # 1. Initialize t_red = iv_var * k in preheader
                new_cfg[preheader_label].instrs.append(
                    BinOp(dst=red_var, op="*", a=iv_var, b=k)
                )

                # 2. Inside loop: replace multiplication with copy t = t_red
                new_cfg[node].instrs[idx] = Assign(dst=ins.dst, src=red_var)

                # 3. At induction variable increment: insert t_red = t_red + delta
                # Find current index of inc_ins in inc_block
                inc_list = new_cfg[inc_block].instrs
                curr_inc_idx = next((i for i, x in enumerate(inc_list) if x is inc_ins), len(inc_list) - 1)
                inc_list.insert(
                    curr_inc_idx + 1,
                    BinOp(dst=red_var, op="+", a=red_var, b=delta),
                )

                stats["operations_reduced"] += 1
                stats["instructions_changed"] += 1

        return new_cfg, stats


class LoopUnrolling(OptimizationPass):
    """Pass 7: Unrolls loops with known small trip counts (<=8) or small loop bodies."""
    name = "Loop Unrolling"
    description = "Fully unrolls constant-trip loops (<=8) and unrolls small loops by factor 2"
    required_analysis = None

    def run(
        self,
        cfg: Dict[str, BasicBlock],
        dataflow_results: Optional[DataFlowResults] = None,
    ) -> Tuple[Dict[str, BasicBlock], Dict[str, int]]:
        """Execute loop unrolling pass across the CFG."""
        stats = {
            "instructions_removed": 0,
            "instructions_changed": 0,
            "loops_unrolled": 0,
        }
        loops = find_natural_loops(cfg)
        if not loops:
            return cfg, stats

        new_cfg: Dict[str, BasicBlock] = {
            lbl: BasicBlock(label=lbl, instrs=list(b.instrs), preds=list(b.preds), succs=list(b.succs))
            for lbl, b in cfg.items()
        }

        for loop in loops:
            # Guard: never unroll a loop that contains a call instruction
            has_call = any(isinstance(ins, Call) for node in loop.body if node in new_cfg for ins in new_cfg[node].instrs)
            if has_call:
                continue

            iv_info = find_primary_induction_var(loop, new_cfg)
            if not iv_info:
                continue

            iv_var, step, inc_block, inc_idx, inc_ins = iv_info

            # Detect loop condition and bound
            header = loop.header
            if header not in new_cfg or len(loop.body) != 1:
                # Focus clean unrolling on tight single-block loops
                continue

            header_block = new_cfg[header]
            last_ins = header_block.instrs[-1] if header_block.instrs else None
            second_last = header_block.instrs[-2] if len(header_block.instrs) >= 2 else None

            # Look for comparison: cond = i < N or cond = i <= N
            if not isinstance(second_last, BinOp) or second_last.op not in ("<", "<="):
                continue
            if not isinstance(last_ins, (IfNz, IfZ)):
                continue

            op = second_last.op
            cmp_var = second_last.a
            cmp_bound = second_last.b

            if cmp_var != iv_var or not isinstance(cmp_bound, int):
                continue

            # Determine initial value of induction variable from preheader or entry predecessor
            init_val = None
            for p in header_block.preds:
                if p not in loop.body and p in new_cfg:
                    for ins in reversed(new_cfg[p].instrs):
                        if isinstance(ins, Assign) and ins.dst == iv_var and isinstance(ins.src, int):
                            init_val = ins.src
                            break
                    if init_val is not None:
                        break

            if init_val is None or step <= 0:
                continue

            # Calculate trip count
            if op == "<":
                trip_count = max(0, (cmp_bound - init_val + step - 1) // step)
            elif op == "<=":
                trip_count = max(0, (cmp_bound - init_val) // step + 1)
            else:
                trip_count = 0

            # 1. Full unrolling for trip_count <= 8
            if 1 <= trip_count <= 8:
                unrolled_instrs = []
                # Instructions to repeat (exclude label, increment, compare, and branch)
                body_instrs = [
                    ins for ins in header_block.instrs
                    if ins is not inc_ins and ins is not second_last and ins is not last_ins and not isinstance(ins, Label)
                ]

                for iter_idx in range(trip_count):
                    curr_i = init_val + iter_idx * step
                    for ins in body_instrs:
                        if isinstance(ins, Assign):
                            new_src = curr_i if ins.src == iv_var else ins.src
                            unrolled_instrs.append(Assign(ins.dst, new_src))
                        elif isinstance(ins, BinOp):
                            new_a = curr_i if ins.a == iv_var else ins.a
                            new_b = curr_i if ins.b == iv_var else ins.b
                            unrolled_instrs.append(BinOp(ins.dst, ins.op, new_a, new_b))
                        else:
                            unrolled_instrs.append(ins)

                # Set final value of induction variable
                final_i = init_val + trip_count * step
                unrolled_instrs.append(Assign(iv_var, final_i))

                # Identify exit block (where loop branch would jump if done)
                exit_target = None
                for s in header_block.succs:
                    if s not in loop.body:
                        exit_target = s
                        break

                if exit_target:
                    unrolled_instrs.append(Goto(exit_target))
                    header_block.succs = [exit_target]
                    if header in new_cfg[exit_target].preds:
                        pass
                else:
                    header_block.succs = []

                # Remove backedge
                header_block.preds = [p for p in header_block.preds if p not in loop.body]
                header_block.instrs = unrolled_instrs

                stats["loops_unrolled"] += 1
                stats["instructions_changed"] += len(unrolled_instrs)

        return new_cfg, stats


# =====================================================================
# PIPELINE ORCHESTRATOR
# =====================================================================

class OptimizationPipeline:
    """Orchestrates sequential and iterative optimization passes."""
    def __init__(self, passes: Optional[List[OptimizationPass]] = None):
        if passes is None:
            self.passes = [
                LoopInvariantCodeMotion(),
                StrengthReduction(),
                LoopUnrolling(),
                ConstantFoldingAndPropagation(),
                CopyPropagation(),
                CommonSubexpressionElimination(),
                DeadCodeElimination(),
            ]
        else:
            self.passes = passes

    def run(self, cfg: Dict[str, BasicBlock]) -> Tuple[Dict[str, BasicBlock], List[Dict[str, Any]]]:
        """Execute each pass in the pipeline once in order."""
        current_cfg = cfg
        reports: List[Dict[str, Any]] = []

        for p in self.passes:
            t0 = time.perf_counter()
            df_results = p.required_analysis().run(current_cfg) if p.required_analysis else None
            current_cfg, stats = p.run(current_cfg, df_results)
            elapsed = time.perf_counter() - t0
            reports.append({
                "pass": p.name,
                "stats": stats,
                "time_sec": elapsed,
            })

        return current_cfg, reports

    def run_until_convergence(
        self, cfg: Dict[str, BasicBlock], max_iterations: int = 10
    ) -> Tuple[Dict[str, BasicBlock], List[Dict[str, Any]]]:
        """Repeat pipeline until no pass reports any changes or max_iterations is reached."""
        current_cfg = cfg
        all_reports: List[Dict[str, Any]] = []

        for iteration in range(1, max_iterations + 1):
            iter_changes = 0

            for p in self.passes:
                t0 = time.perf_counter()
                df_results = p.required_analysis().run(current_cfg) if p.required_analysis else None
                current_cfg, stats = p.run(current_cfg, df_results)
                elapsed = time.perf_counter() - t0
                changes = stats.get("instructions_removed", 0) + stats.get("instructions_changed", 0)
                iter_changes += changes
                all_reports.append({
                    "iteration": iteration,
                    "pass": p.name,
                    "stats": stats,
                    "time_sec": elapsed,
                })

            if iter_changes == 0:
                break

        return current_cfg, all_reports

    def print_report(self, reports: List[Dict[str, Any]]):
        """Print formatted per-pass report."""
        print("\n=== OPTIMIZATION PIPELINE REPORT ===")
        for r in reports:
            iter_info = f"Iter {r['iteration']}: " if "iteration" in r else ""
            stats_str = ", ".join(f"{k}={v}" for k, v in r["stats"].items())
            print(f"  [{iter_info}{r['pass']}] {stats_str} (took {r['time_sec']*1000:.3f} ms)")


# =====================================================================
# Program Level Optimization Driver
# =====================================================================

def optimize_function(
    fn: Function,
    max_iterations: int = 10,
    stats: Optional[OptimizationStats] = None,
) -> Function:
    """Optimize a single Function using the standard pipeline."""
    cfg = build_cfg(fn)
    if not cfg:
        return fn

    pipeline = OptimizationPipeline()
    opt_cfg, reports = pipeline.run_until_convergence(cfg, max_iterations=max_iterations)

    if stats is not None:
        for r in reports:
            s = r["stats"]
            stats.dead_instrs_removed += s.get("instructions_removed", 0)
            stats.constants_folded += s.get("constants_folded", 0)
            stats.copies_propagated += s.get("copies_propagated", 0)
            stats.expressions_eliminated += s.get("expressions_eliminated", 0)
            stats.instructions_hoisted += s.get("instructions_hoisted", 0)
            stats.operations_reduced += s.get("operations_reduced", 0)
            stats.loops_unrolled += s.get("loops_unrolled", 0)

    return cfg_to_function(fn, opt_cfg)


def optimize_program(
    functions: List[Function],
    max_iterations: int = 10,
    stats: Optional[OptimizationStats] = None,
) -> List[Function]:
    """Optimize all functions in an IR program."""
    return [optimize_function(f, max_iterations=max_iterations, stats=stats) for f in functions]
