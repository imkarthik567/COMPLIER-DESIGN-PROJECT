"""CVM AI Pass Selector.

Phase 8 of CVM optimizing compiler backend.
Analyzes program characteristics from the IR and CFG, scores each optimization pass
using an explainable heuristic AI model with feedback learning from pass_history.json,
and determines the optimal sequence of compiler passes to apply.

Provides:
- ProgramFeatures: Dataclass capturing structural, expression, variable, and loop features
- ProgramFeatureExtractor: Analyzes IR/CFG to extract program profile metrics
- PassDecision: Decision and score per pass with explainable reason
- PassScoringModel: Rule-based heuristic scoring engine with history-adjusted weights
- PassSequenceGenerator: Topological orderer respecting strict compiler pipeline constraints
- AIPassSelector: High-level selector orchestrating the entire workflow
- History feedback loop: Saves outcomes to and loads historical outcomes from pass_history.json
- run_ai_pipeline: Executes selected passes cleanly
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import codegen
import optimizer
from optimizer import (
    BasicBlock,
    CommonSubexpressionElimination,
    ConstantFoldingAndPropagation,
    CopyPropagation,
    DeadCodeElimination,
    LoopInvariantCodeMotion,
    LoopUnrolling,
    OptimizationPipeline,
    OptimizationStats,
    StrengthReduction,
    build_cfg,
    cfg_to_function,
)
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
import loops
import regalloc
import scheduler
import vectorizer


# =====================================================================
# STEP 1: Program Feature Extractor
# =====================================================================

@dataclass
class ProgramFeatures:
    """Rich feature vector characterizing an IR program."""
    # Structural features
    loop_count: int = 0
    max_loop_nesting_depth: int = 0
    has_inner_loops: bool = False
    avg_block_size: float = 0.0
    branch_density: float = 0.0
    call_density: float = 0.0
    function_count: int = 0

    # Expression features
    const_literal_fraction: float = 0.0
    copy_instruction_fraction: float = 0.0
    arithmetic_intensity: float = 0.0
    array_access_count: int = 0
    multiply_divide_count: int = 0

    # Variable features
    max_live_variables: int = 0
    variable_reuse_rate: float = 0.0
    dead_variable_fraction: float = 0.0
    cse_opportunity_count: int = 0

    # Loop features
    loops_with_array_access: int = 0
    loops_with_invariant_code: int = 0
    loops_with_induction_vars: int = 0
    loops_with_known_count: int = 0

    # Totals
    total_instructions: int = 0
    total_blocks: int = 0
    total_operands: int = 0
    arithmetic_instructions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert program features dataclass to a standard dictionary."""
        return asdict(self)


class ProgramFeatureExtractor:
    """Extracts structural, expression, variable, and loop features from CVM IR."""

    def extract(self, functions: List[Function]) -> ProgramFeatures:
        """Extract all static program features from a list of IR functions."""
        if not functions:
            return ProgramFeatures()

        feat = ProgramFeatures(function_count=len(functions))

        total_instrs_all: List[Any] = []
        total_blocks_count = 0
        all_loops: List[loops.NaturalLoop] = []
        max_depth_all = 0

        const_operands = 0
        total_operands = 0
        copy_instrs = 0
        arith_instrs = 0
        mem_instrs = 0
        branch_instrs = 0
        call_instrs = 0
        array_accesses = 0
        mul_div_count = 0

        max_live_across_funcs = 0
        total_uses = 0
        total_defs = 0
        dead_defs = 0
        cse_opps = 0

        loops_with_arr = 0
        loops_with_invar = 0
        loops_with_iv = 0
        loops_with_known = 0

        for fn in functions:
            cfg = build_cfg(fn)
            if not cfg:
                continue

            total_blocks_count += len(cfg)
            fn_instrs = [ins for b in cfg.values() for ins in b.instrs if not isinstance(ins, Label)]
            total_instrs_all.extend(fn_instrs)

            # --- Structural: Loops & Nesting ---
            fn_loops = loops.find_natural_loops(cfg)
            all_loops.extend(fn_loops)
            if fn_loops:
                nesting_tree = loops.build_loop_nesting_tree(fn_loops)
                for l in fn_loops:
                    d = nesting_tree.nesting_depth(l)
                    if d > max_depth_all:
                        max_depth_all = d

            # --- Variable analysis: Liveness & Def-Use ---
            liveness = dataflow.LivenessAnalysis().run(cfg)

            for b in cfg.values():
                curr_live = set(liveness.out_sets.get(b.label, set()))
                max_live_across_funcs = max(max_live_across_funcs, len(curr_live))

                for ins in reversed(b.instrs):
                    if isinstance(ins, Label):
                        continue

                    # Operands & expressions
                    dst = dataflow._get_def_var(ins)
                    reads = dataflow._get_read_vars(ins)

                    total_uses += len(reads)
                    if dst:
                        total_defs += 1
                        # Dead code detection: dst defined but not live after definition
                        if dst not in curr_live and not isinstance(ins, Call):
                            dead_defs += 1
                        curr_live.discard(dst)

                    for r in reads:
                        curr_live.add(r)
                    max_live_across_funcs = max(max_live_across_funcs, len(curr_live))

                    # Instruction type classification
                    if isinstance(ins, Assign):
                        copy_instrs += 1
                        total_operands += 1
                        if isinstance(ins.src, int):
                            const_operands += 1
                    elif isinstance(ins, BinOp):
                        total_operands += 2
                        if isinstance(ins.a, int):
                            const_operands += 1
                        if isinstance(ins.b, int):
                            const_operands += 1
                        if ins.op in ("+", "-", "*", "/", "%"):
                            arith_instrs += 1
                        if ins.op in ("*", "/", "%"):
                            mul_div_count += 1
                    elif isinstance(ins, (VAdd, VSub, VMul, VDiv)):
                        arith_instrs += 1
                        if isinstance(ins, (VMul, VDiv)):
                            mul_div_count += 1
                    elif isinstance(ins, (LoadIndexed, StoreIndexed, VLoad, VStore)):
                        array_accesses += 1
                        mem_instrs += 1
                        if isinstance(ins, (LoadIndexed, StoreIndexed)):
                            if isinstance(ins.index, int):
                                const_operands += 1
                                total_operands += 1
                    elif isinstance(ins, NewArray):
                        mem_instrs += 1
                        if isinstance(ins.size, int):
                            const_operands += 1
                            total_operands += 1
                    elif isinstance(ins, (IfZ, IfNz)):
                        branch_instrs += 1
                        total_operands += 1
                        if isinstance(ins.cond, int):
                            const_operands += 1
                    elif isinstance(ins, Goto):
                        branch_instrs += 1
                    elif isinstance(ins, Call):
                        call_instrs += 1
                        for a in ins.args:
                            total_operands += 1
                            if isinstance(a, int):
                                const_operands += 1
                    elif isinstance(ins, Print):
                        total_operands += 1
                        if isinstance(ins.arg, int):
                            const_operands += 1
                    elif isinstance(ins, Return):
                        if ins.arg is not None:
                            total_operands += 1
                            if isinstance(ins.arg, int):
                                const_operands += 1

            # --- CSE Opportunities ---
            expr_counts: Dict[Tuple[str, Any, Any], int] = {}
            for ins in fn_instrs:
                if isinstance(ins, BinOp):
                    k = (ins.op, ins.a, ins.b)
                    expr_counts[k] = expr_counts.get(k, 0) + 1
            cse_opps += sum(c - 1 for c in expr_counts.values() if c > 1)

            # --- Loop characteristics ---
            for loop_obj in fn_loops:
                loop_nodes = loop_obj.body
                loop_instrs = [ins for node in loop_nodes if node in cfg for ins in cfg[node].instrs]

                # Array access in loop
                if any(isinstance(ins, (LoadIndexed, StoreIndexed, VLoad, VStore)) for ins in loop_instrs):
                    loops_with_arr += 1

                # Induction variables
                iv_info = loops.find_primary_induction_var(loop_obj, cfg)
                if iv_info is not None:
                    loops_with_iv += 1
                    iv_var, step, inc_block, inc_idx, inc_ins = iv_info

                    # Statically known trip count check
                    bound_known = False
                    for exit_block_lbl in loop_obj.exit_blocks:
                        if exit_block_lbl in cfg:
                            for ins in cfg[exit_block_lbl].instrs:
                                if isinstance(ins, (IfZ, IfNz)):
                                    cond_var = ins.cond
                                    for b_ins in cfg[exit_block_lbl].instrs:
                                        if isinstance(b_ins, BinOp) and b_ins.dst == cond_var:
                                            bound = b_ins.b if b_ins.a == iv_var else b_ins.a
                                            if isinstance(bound, int):
                                                bound_known = True
                    if bound_known:
                        loops_with_known += 1

                # Loop invariant expressions
                defs_in_loop = set()
                for ins in loop_instrs:
                    d = dataflow._get_def_var(ins)
                    if d:
                        defs_in_loop.add(d)

                has_invar = False
                for ins in loop_instrs:
                    if isinstance(ins, BinOp):
                        # Both operands defined outside the loop or constant
                        op_a_inv = isinstance(ins.a, int) or (isinstance(ins.a, str) and ins.a not in defs_in_loop)
                        op_b_inv = isinstance(ins.b, int) or (isinstance(ins.b, str) and ins.b not in defs_in_loop)
                        if op_a_inv and op_b_inv:
                            has_invar = True
                            break
                if has_invar:
                    loops_with_invar += 1

        total_ins_count = len(total_instrs_all)
        feat.total_instructions = total_ins_count
        feat.total_blocks = total_blocks_count
        feat.total_operands = total_operands
        feat.arithmetic_instructions = arith_instrs

        # Structural
        feat.loop_count = len(all_loops)
        feat.max_loop_nesting_depth = max_depth_all
        feat.has_inner_loops = max_depth_all > 1
        feat.avg_block_size = total_ins_count / max(total_blocks_count, 1)
        feat.branch_density = branch_instrs / max(total_ins_count, 1)
        feat.call_density = call_instrs / max(total_ins_count, 1)

        # Expressions
        feat.const_literal_fraction = const_operands / max(total_operands, 1)
        feat.copy_instruction_fraction = copy_instrs / max(total_ins_count, 1)
        feat.arithmetic_intensity = arith_instrs / max(mem_instrs, 1)
        feat.array_access_count = array_accesses
        feat.multiply_divide_count = mul_div_count

        # Variables
        feat.max_live_variables = max_live_across_funcs
        feat.variable_reuse_rate = total_uses / max(total_defs, 1)
        feat.dead_variable_fraction = dead_defs / max(total_defs, 1)
        feat.cse_opportunity_count = cse_opps

        # Loops
        feat.loops_with_array_access = loops_with_arr
        feat.loops_with_invariant_code = loops_with_invar
        feat.loops_with_induction_vars = loops_with_iv
        feat.loops_with_known_count = loops_with_known

        return feat


# =====================================================================
# STEP 2 & 4: Pass Scoring Model & Feedback Loop
# =====================================================================

CANONICAL_ORDER = [
    "LICM",
    "StrengthReduction",
    "LoopUnrolling",
    "ConstantFolding",
    "CopyPropagation",
    "CSE",
    "DeadCodeElimination",
    "Vectorizer",
    "RegisterAllocation",
    "InstructionScheduler",
]

DEFAULT_HISTORY_FILE = "pass_history.json"


@dataclass
class PassDecision:
    """Scoring decision and justification for a single optimization pass."""
    name: str
    score: float
    raw_score: float
    reason: str
    selected: bool
    is_prerequisite: bool = False


def load_history(history_file: str = DEFAULT_HISTORY_FILE) -> List[Dict[str, Any]]:
    """Load historical pass outcomes from JSON."""
    if not os.path.exists(history_file):
        return []
    try:
        with open(history_file, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def record_history(
    features: Dict[str, Any],
    passes_run: List[str],
    instructions_before: int,
    instructions_after: int,
    speedup: float = 1.0,
    history_file: str = DEFAULT_HISTORY_FILE,
) -> None:
    """Record a run outcome into pass_history.json."""
    history = load_history(history_file)
    entry = {
        "timestamp": time.time(),
        "features": features,
        "passes_run": list(passes_run),
        "instructions_before": instructions_before,
        "instructions_after": instructions_after,
        "speedup": round(speedup, 4),
    }
    history.append(entry)
    # Retain most recent 200 runs
    if len(history) > 200:
        history = history[-200:]
    try:
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)
    except OSError:
        pass


class PassScoringModel:
    """Rule-based heuristic scoring engine with history feedback learning."""

    N_REGS = 8  # Architecture virtual register limit

    def score(
        self,
        features: ProgramFeatures,
        history_file: str = DEFAULT_HISTORY_FILE,
    ) -> Dict[str, PassDecision]:
        """Compute expected benefit scores [0.0, 1.0] for all optimization passes."""
        tot = max(features.total_instructions, 1)
        loops_cnt = max(features.loop_count, 1)

        # -------------------------------------------------------------
        # Heuristic Rule-Based Scoring Formulas
        # -------------------------------------------------------------
        # 1. Dead Code Elimination
        raw_dce = features.dead_variable_fraction * 0.9 + features.copy_instruction_fraction * 0.3
        dce_reason = f"estimated {int(features.dead_variable_fraction * 100)}% dead variables"

        # 2. Constant Folding & Propagation
        raw_const = features.const_literal_fraction * 1.0
        const_reason = f"{int(features.const_literal_fraction * 100)}% literal constants detected"

        # 3. Copy Propagation
        raw_copy = features.copy_instruction_fraction * 0.85
        copy_reason = f"{int(features.copy_instruction_fraction * 100)}% copy instructions"

        # 4. Common Subexpression Elimination
        raw_cse = (features.cse_opportunity_count / tot) * 0.7
        if features.cse_opportunity_count > 0 and raw_cse < 0.2:
            raw_cse = min(0.7, 0.25 + (features.cse_opportunity_count / tot) * 0.5)
        cse_reason = (
            f"{features.cse_opportunity_count} repeated expressions identified"
            if features.cse_opportunity_count > 0
            else "low repeated expression count"
        )

        # 5. Loop Invariant Code Motion (LICM)
        if features.loop_count > 0 and features.loops_with_invariant_code > 0:
            raw_licm = (features.loops_with_invariant_code / loops_cnt) * features.loop_count * 0.5
            raw_licm = min(1.0, max(0.4, raw_licm))
            licm_reason = f"{features.loops_with_invariant_code} loops with invariant expressions"
        else:
            raw_licm = 0.0
            licm_reason = "no loops with invariant expressions"

        # 6. Strength Reduction
        if features.loops_with_induction_vars > 0 and features.multiply_divide_count > 0:
            raw_sr = min(1.0, (features.multiply_divide_count * features.loops_with_induction_vars) * 0.3 + 0.3)
            sr_reason = f"{features.multiply_divide_count} mul/div ops in induction loops"
        else:
            raw_sr = 0.0
            sr_reason = "no mul/div operations in induction loops"

        # 7. Loop Unrolling
        if features.loops_with_known_count > 0:
            raw_unroll = features.loops_with_known_count * (1.0 - features.call_density) * 0.4
            raw_unroll = min(1.0, max(0.3, raw_unroll))
            unroll_reason = f"{features.loops_with_known_count} loops with known trip count"
        else:
            raw_unroll = 0.0
            unroll_reason = "no loops with known trip count"

        # 8. Vectorizer
        if features.loops_with_array_access > 0:
            raw_vec = features.loops_with_array_access * (1.0 - features.branch_density) * 0.8
            raw_vec = min(1.0, max(0.5, raw_vec))
            vec_reason = f"{features.loops_with_array_access} loops with array access, low branches"
        else:
            raw_vec = 0.0
            vec_reason = "no array-access loops detected"

        # 9. Register Allocation
        if features.max_live_variables >= self.N_REGS:
            raw_reg = min(1.0, max(0.85, (features.max_live_variables / self.N_REGS) * 0.9))
            reg_reason = f"high variable pressure ({features.max_live_variables} live vars > 8)"
        elif features.max_live_variables > 0:
            raw_reg = min(1.0, (features.max_live_variables / self.N_REGS) * 0.9)
            reg_reason = f"{features.max_live_variables} live variables benefit from virtual registers"
        else:
            raw_reg = 0.1
            reg_reason = "minimal variable live ranges"

        # 10. Instruction Scheduling
        raw_sched = features.arithmetic_intensity * (1.0 - features.call_density) * 0.5
        raw_sched = min(1.0, max(0.0, raw_sched))
        sched_reason = (
            f"moderate arithmetic intensity ({features.arithmetic_intensity:.1f})"
            if raw_sched >= 0.2
            else "low arithmetic intensity / stalls"
        )

        heuristic_scores: Dict[str, Tuple[float, str]] = {
            "LICM": (min(1.0, max(0.0, raw_licm)), licm_reason),
            "StrengthReduction": (min(1.0, max(0.0, raw_sr)), sr_reason),
            "LoopUnrolling": (min(1.0, max(0.0, raw_unroll)), unroll_reason),
            "ConstantFolding": (min(1.0, max(0.0, raw_const)), const_reason),
            "CopyPropagation": (min(1.0, max(0.0, raw_copy)), copy_reason),
            "CSE": (min(1.0, max(0.0, raw_cse)), cse_reason),
            "DeadCodeElimination": (min(1.0, max(0.0, raw_dce)), dce_reason),
            "Vectorizer": (min(1.0, max(0.0, raw_vec)), vec_reason),
            "RegisterAllocation": (min(1.0, max(0.0, raw_reg)), reg_reason),
            "InstructionScheduler": (min(1.0, max(0.0, raw_sched)), sched_reason),
        }

        # -------------------------------------------------------------
        # STEP 4: Feedback Loop Adjustment from pass_history.json
        # -------------------------------------------------------------
        history = load_history(history_file)
        decisions: Dict[str, PassDecision] = {}

        for pass_name, (h_score, reason) in heuristic_scores.items():
            if not history:
                final_score = h_score
            else:
                pass_records = [r for r in history if pass_name in r.get("passes_run", [])]
                if not pass_records:
                    final_score = h_score
                else:
                    benefits = []
                    for r in pass_records:
                        before = r.get("instructions_before", 1)
                        after = r.get("instructions_after", before)
                        red = max(0.0, (before - after) / max(before, 1))
                        spd = max(1.0, r.get("speedup", 1.0))
                        b = min(1.0, red * 1.5 + (spd - 1.0) * 0.25)
                        benefits.append(b)
                    hist_avg = sum(benefits) / len(benefits)
                    # adjusted_score = 0.7 * heuristic_score + 0.3 * historical_avg_benefit
                    final_score = 0.7 * h_score + 0.3 * hist_avg

            final_score = min(1.0, max(0.0, final_score))
            decisions[pass_name] = PassDecision(
                name=pass_name,
                score=round(final_score, 2),
                raw_score=round(h_score, 2),
                reason=reason,
                selected=False,  # updated by sequence generator
            )

        return decisions


# =====================================================================
# STEP 3: Pass Sequence Generator & Topological Constraints
# =====================================================================

class PassSequenceGenerator:
    """Generates the optimal, topologically sorted pass execution sequence."""

    THRESHOLD = 0.2

    # Prerequisite dependencies: pass -> required prerequisite passes
    PREREQUISITES: Dict[str, List[str]] = {
        "InstructionScheduler": ["RegisterAllocation"],
    }

    def generate(
        self,
        decisions: Dict[str, PassDecision],
        threshold: float = THRESHOLD,
    ) -> Tuple[List[str], List[str]]:
        """Select candidate passes with score >= threshold and order topologically."""
        selected_set: Set[str] = set()

        for name, dec in decisions.items():
            if dec.score >= threshold:
                selected_set.add(name)

        # Include required prerequisites at minimum strength if target pass scored high
        for target_pass, prereqs in self.PREREQUISITES.items():
            if target_pass in selected_set:
                for prereq in prereqs:
                    if prereq not in selected_set:
                        selected_set.add(prereq)
                        decisions[prereq].is_prerequisite = True

        # Sort according to canonical compiler pipeline ordering constraints
        ordered_selected = [p for p in CANONICAL_ORDER if p in selected_set]
        skipped = [p for p in CANONICAL_ORDER if p not in selected_set]

        # Update decision objects
        for name, dec in decisions.items():
            dec.selected = name in selected_set

        return ordered_selected, skipped


# =====================================================================
# STEP 5: Explainability Report & High-Level Selector
# =====================================================================

@dataclass
class AISelectorResult:
    """Complete output of AI Pass Selector including report formatting."""
    features: ProgramFeatures
    decisions: Dict[str, PassDecision]
    selected_passes: List[str]
    skipped_passes: List[str]

    def format_report(self) -> str:
        """Format AI pass selection features, scores, and decisions into a human-readable report."""
        tot_ins = max(self.features.total_instructions, 1)
        arith_pct = int((self.features.arithmetic_instructions / tot_ins) * 100)
        const_pct = int(self.features.const_literal_fraction * 100)

        lines = [
            "AI Pass Selector Report",
            "=======================",
            "Program Profile:",
            (
                f"  - {self.features.loop_count} loops (max depth {self.features.max_loop_nesting_depth}), "
                f"{arith_pct}% arithmetic, {const_pct}% constants, "
                f"{self.features.loops_with_array_access} array-access loops"
            ),
            "",
            "Selected Passes (in order):",
        ]

        if not self.selected_passes:
            lines.append("  (none - all passes scored below threshold)")
        else:
            for idx, p in enumerate(self.selected_passes, 1):
                dec = self.decisions[p]
                score_str = f"score={dec.score:.2f}"
                pre_tag = " [prerequisite]" if dec.is_prerequisite else ""
                lines.append(f"  {idx}. {p:<22} {score_str:<12} reason=\"{dec.reason}\"{pre_tag}")

        lines.extend(["", "Skipped Passes:"])
        if not self.skipped_passes:
            lines.append("  (none - all passes selected)")
        else:
            for p in self.skipped_passes:
                dec = self.decisions[p]
                score_str = f"score={dec.score:.2f}"
                lines.append(f"  - {p:<22} {score_str:<12} reason=\"{dec.reason}\"")

        return "\n".join(lines)

    def report(self) -> str:
        """Alias for format_report to return full human-readable decision report."""
        return self.format_report()


class AIPassSelector:
    """Orchestrates feature extraction, scoring, ordering, and report generation."""

    def __init__(self, history_file: str = DEFAULT_HISTORY_FILE):
        self.extractor = ProgramFeatureExtractor()
        self.scorer = PassScoringModel()
        self.generator = PassSequenceGenerator()
        self.history_file = history_file

    def select(
        self,
        functions: List[Function],
        threshold: float = PassSequenceGenerator.THRESHOLD,
    ) -> AISelectorResult:
        """Analyze functions and select the optimal sequence of passes."""
        features = self.extractor.extract(functions)
        decisions = self.scorer.score(features, history_file=self.history_file)
        selected, skipped = self.generator.generate(decisions, threshold=threshold)
        return AISelectorResult(
            features=features,
            decisions=decisions,
            selected_passes=selected,
            skipped_passes=skipped,
        )


def ai_select_passes(
    functions: List[Function],
    threshold: float = PassSequenceGenerator.THRESHOLD,
    history_file: str = DEFAULT_HISTORY_FILE,
) -> AISelectorResult:
    """Convenience functional interface for AI pass selection."""
    selector = AIPassSelector(history_file=history_file)
    return selector.select(functions, threshold=threshold)


# =====================================================================
# Pipeline Runner for AI-Selected Passes
# =====================================================================

PASS_TO_CLASS = {
    "LICM": LoopInvariantCodeMotion,
    "StrengthReduction": StrengthReduction,
    "LoopUnrolling": LoopUnrolling,
    "ConstantFolding": ConstantFoldingAndPropagation,
    "CopyPropagation": CopyPropagation,
    "CSE": CommonSubexpressionElimination,
    "DeadCodeElimination": DeadCodeElimination,
}


def run_ai_pipeline(
    functions: List[Function],
    selected_passes: List[str],
) -> Tuple[List[Function], Optional[Dict[str, Any]], Any, Any, Any, OptimizationStats]:
    """Execute the exact sequence of passes selected by AI."""
    stats = OptimizationStats()

    # 1. Scalar & loop optimizations
    opt_passes = [
        PASS_TO_CLASS[p]()
        for p in CANONICAL_ORDER
        if p in selected_passes and p in PASS_TO_CLASS
    ]

    if opt_passes:
        optimized_functions = []
        for fn in functions:
            cfg = build_cfg(fn)
            if not cfg:
                optimized_functions.append(fn)
                continue
            pipeline = OptimizationPipeline(passes=opt_passes)
            opt_cfg, reports = pipeline.run_until_convergence(cfg)
            for r in reports:
                s = r.get("stats", {})
                stats.dead_instrs_removed += s.get("instructions_removed", 0)
                stats.constants_folded += s.get("constants_folded", 0)
                stats.copies_propagated += s.get("copies_propagated", 0)
                stats.expressions_eliminated += s.get("expressions_eliminated", 0)
                stats.instructions_hoisted += s.get("instructions_hoisted", 0)
                stats.operations_reduced += s.get("operations_reduced", 0)
                stats.loops_unrolled += s.get("loops_unrolled", 0)
            optimized_functions.append(cfg_to_function(fn, opt_cfg))
        functions = optimized_functions

    # 2. Vectorizer
    vec_stats = None
    if "Vectorizer" in selected_passes:
        functions, vec_stats = vectorizer.vectorize_program(functions)

    # 3. Register Allocation
    reg_map = None
    reg_stats = None
    if "RegisterAllocation" in selected_passes:
        reg_map, functions, reg_stats = regalloc.allocate_registers_for_program(functions)

    # 4. Instruction Scheduling
    sched_stats = None
    if "InstructionScheduler" in selected_passes:
        functions, sched_stats = scheduler.schedule_program(functions)

    return functions, reg_map, reg_stats, sched_stats, vec_stats, stats
