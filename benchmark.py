"""CVM Dual-Compilation Benchmark and Comparison Engine.

Runs an IR program twice:
  Run 1 (BASELINE):  parse IR -> build CFG -> codegen directly (no opt, no regalloc)
  Run 2 (OPTIMIZED): parse IR -> build CFG -> AI pass selector -> all selected passes ->
                     regalloc -> scheduler -> codegen -> verifier

Measures static, dynamic, and derived metrics, verifies output equivalence,
and outputs formatted Text, JSON, and Markdown reports.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Enable utf-8 terminal output on Windows if possible
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import ordering precaution: optimizer before dataflow
import optimizer
import dataflow
import vectorizer
import ir
import isa
from isa import Op
import codegen
import verifier
from verifier import Verifier
import vm
from vm import VM, VMError
import regalloc
import scheduler
import loops
import ai_selector


class CorrectnessError(Exception):
    """Raised when optimized program output does not match baseline output."""
    pass


class ProfilingVM(VM):
    """Execution VM that tracks dynamic metrics without printing to stdout."""

    def __init__(self, code: List[Any], out=None, **kwargs):
        self.recorded_output: List[Any] = []
        capture_out = (lambda v: self.recorded_output.append(v)) if out is None else out
        super().__init__(code, trace=True, out=capture_out, **kwargs)
        self.peak_stack_depth: int = 0
        self.memory_accesses: int = 0
        self.call_count: int = 0

    def reset(self) -> None:
        """Reset execution state and dynamic metric counters."""
        super().reset()
        self.peak_stack_depth = 0
        self.memory_accesses = 0
        self.call_count = 0
        self.recorded_output = []

    def _trace(self, ins: Any) -> None:
        """Non-printing trace callback to record execution metrics."""
        # 1. Track peak operand stack depth
        cur_depth = len(self.stack)
        if cur_depth > self.peak_stack_depth:
            self.peak_stack_depth = cur_depth

        # 2. Track memory slot accesses (proxy for cache misses)
        # LOAD, STORE, LOAD_INDEXED, STORE_INDEXED, VLOAD, VSTORE, RSTORE
        if ins.op in (
            Op.LOAD,
            Op.STORE,
            Op.LOAD_INDEXED,
            Op.STORE_INDEXED,
            Op.VLOAD,
            Op.VSTORE,
            Op.RSTORE,
        ):
            self.memory_accesses += 1

        # 3. Track function calls
        if ins.op == Op.CALL:
            self.call_count += 1


@dataclass
class BenchmarkMetrics:
    """Holds all benchmark measurements for baseline vs. optimized compilation."""

    program: str
    timestamp: str
    correctness: bool
    baseline_output: List[Any]
    optimized_output: List[Any]
    selected_passes: List[str]

    # Static Metrics
    instruction_count_before: int
    instruction_count_after: int
    basic_block_count_before: int
    basic_block_count_after: int
    loop_count_before: int
    loop_count_after: int
    dead_code_removed: int
    constants_folded: int
    copies_eliminated: int
    cse_replacements: int
    loops_vectorized: int
    instructions_hoisted_by_LICM: int
    estimated_register_spills: int

    # Dynamic Metrics
    vm_instructions_executed_before: int
    vm_instructions_executed_after: int
    execution_time_ms_before: float
    execution_time_ms_after: float
    operand_stack_peak_depth_before: int
    operand_stack_peak_depth_after: int
    estimated_cache_misses_before: int
    estimated_cache_misses_after: int
    function_call_count_before: int
    function_call_count_after: int

    # Derived Metrics
    static_speedup: float
    dynamic_speedup: float
    wall_speedup: float
    code_size_reduction_percent: float

    # Detailed pass contributions
    pass_contributions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a structured dictionary."""
        return {
            "program": self.program,
            "timestamp": self.timestamp,
            "correctness": self.correctness,
            "selected_passes": list(self.selected_passes),
            "static_metrics": {
                "instruction_count_before": self.instruction_count_before,
                "instruction_count_after": self.instruction_count_after,
                "basic_block_count_before": self.basic_block_count_before,
                "basic_block_count_after": self.basic_block_count_after,
                "loop_count_before": self.loop_count_before,
                "loop_count_after": self.loop_count_after,
                "dead_code_removed": self.dead_code_removed,
                "constants_folded": self.constants_folded,
                "copies_eliminated": self.copies_eliminated,
                "cse_replacements": self.cse_replacements,
                "loops_vectorized": self.loops_vectorized,
                "instructions_hoisted_by_LICM": self.instructions_hoisted_by_LICM,
                "estimated_register_spills": self.estimated_register_spills,
            },
            "dynamic_metrics": {
                "vm_instructions_executed_before": self.vm_instructions_executed_before,
                "vm_instructions_executed_after": self.vm_instructions_executed_after,
                "execution_time_ms_before": round(self.execution_time_ms_before, 3),
                "execution_time_ms_after": round(self.execution_time_ms_after, 3),
                "operand_stack_peak_depth_before": self.operand_stack_peak_depth_before,
                "operand_stack_peak_depth_after": self.operand_stack_peak_depth_after,
                "estimated_cache_misses_before": self.estimated_cache_misses_before,
                "estimated_cache_misses_after": self.estimated_cache_misses_after,
                "function_call_count_before": self.function_call_count_before,
                "function_call_count_after": self.function_call_count_after,
            },
            "derived_metrics": {
                "static_speedup": round(self.static_speedup, 2),
                "dynamic_speedup": round(self.dynamic_speedup, 2),
                "wall_speedup": round(self.wall_speedup, 2),
                "code_size_reduction_percent": round(self.code_size_reduction_percent, 1),
            },
            "pass_contributions": self.pass_contributions,
        }

    def format_text_report(self) -> str:
        """Format metrics as a terminal text box using unicode characters."""
        W = 56
        inner_w = W - 2

        def box_line(content: str = "") -> str:
            # Handle unicode character alignment
            # Truncate if exceeds width, else pad
            if len(content) > inner_w:
                content = content[:inner_w]
            return f"║{content.ljust(inner_w)}║"

        lines = [
            "╔" + "═" * inner_w + "╗",
            box_line("        CVM OPTIMIZATION BENCHMARK REPORT"),
            "╠" + "═" * inner_w + "╣",
        ]
        # Header & Program info
        lines.append(box_line(f"  Program: {self.program}"))
        passes_str = ", ".join(self.selected_passes)
        prefix = "  AI Selected Passes: "
        if len(prefix + passes_str) <= inner_w:
            lines.append(box_line(prefix + passes_str))
        else:
            # Wrap gracefully
            import textwrap
            wrapped = textwrap.wrap(passes_str, width=inner_w - len(prefix))
            for i, part in enumerate(wrapped):
                if i == 0:
                    lines.append(box_line(prefix + part))
                else:
                    lines.append(box_line(" " * len(prefix) + part))

        lines.extend([
            "╠" + "═" * inner_w + "╣",
            box_line("  STATIC ANALYSIS"),
        ])

        # Instruction reduction
        instr_change = (
            f"({self.code_size_reduction_percent:+.1f}%)"
            if self.code_size_reduction_percent <= 0
            else f"(-{self.code_size_reduction_percent:.1f}%)"
        )
        lines.append(
            box_line(
                f"   Instructions:  {self.instruction_count_before} → {self.instruction_count_after}    {instr_change}"
            )
        )

        # Basic blocks
        bb_delta = (
            (self.basic_block_count_after - self.basic_block_count_before)
            / max(self.basic_block_count_before, 1)
            * 100.0
        )
        bb_str = f"({bb_delta:+.1f}%)" if bb_delta >= 0 else f"(-{abs(bb_delta):.1f}%)"
        lines.append(
            box_line(
                f"   Basic Blocks:  {self.basic_block_count_before}  → {self.basic_block_count_after}    {bb_str}"
            )
        )

        if self.dead_code_removed > 0:
            lines.append(box_line(f"   Dead code removed: {self.dead_code_removed} instructions"))
        if self.constants_folded > 0:
            lines.append(box_line(f"   Constants folded:  {self.constants_folded}"))
        if self.instructions_hoisted_by_LICM > 0:
            lines.append(box_line(f"   LICM hoisted:      {self.instructions_hoisted_by_LICM}"))
        if self.loops_vectorized > 0:
            lines.append(box_line(f"   Loops vectorized:  {self.loops_vectorized}"))

        lines.extend([
            "╠" + "═" * inner_w + "╣",
            box_line("  DYNAMIC EXECUTION"),
        ])

        # VM Steps
        steps_delta = (
            (self.vm_instructions_executed_after - self.vm_instructions_executed_before)
            / max(self.vm_instructions_executed_before, 1)
            * 100.0
        )
        steps_str = f"({steps_delta:+.1f}%)" if steps_delta >= 0 else f"(-{abs(steps_delta):.1f}%)"
        lines.append(
            box_line(
                f"   VM steps executed: {self.vm_instructions_executed_before:,} → {self.vm_instructions_executed_after:,}   {steps_str}"
            )
        )

        # Wall Time
        lines.append(
            box_line(
                f"   Wall time (median): {self.execution_time_ms_before:.2f}ms → {self.execution_time_ms_after:.2f}ms ({self.wall_speedup:.2f}× faster)"
            )
        )

        # Stack Peak Depth
        lines.append(
            box_line(
                f"   Stack peak depth:  {self.operand_stack_peak_depth_before} → {self.operand_stack_peak_depth_after}"
            )
        )

        # Cache misses proxy
        lines.append(
            box_line(
                f"   Memory slot ops:   {self.estimated_cache_misses_before} → {self.estimated_cache_misses_after}"
            )
        )

        lines.extend([
            "╠" + "═" * inner_w + "╣",
            box_line("  PASS-BY-PASS CONTRIBUTION"),
        ])

        total_gain = self.instruction_count_before - self.instruction_count_after

        for p, info in self.pass_contributions.items():
            if isinstance(info, dict):
                instrs = info.get("instructions_saved", 0)
                if instrs > 0:
                    if total_gain > 0:
                        pct = (instrs / total_gain) * 100.0
                        lines.append(box_line(f"   {p:17s} −{instrs} instr ({pct:.1f}% gain)"))
                    else:
                        lines.append(box_line(f"   {p:17s} −{instrs} instr"))
                elif p == "RegisterAllocation":
                    spills = info.get("spills", 0)
                    regs = info.get("regs_allocated", 0)
                    lines.append(box_line(f"   {p:18s} {spills} spills, {regs} virtual regs"))
                elif p == "InstructionScheduler":
                    stalls = info.get("stalls_saved", 0)
                    lines.append(box_line(f"   {p:18s} −{stalls} pipeline stalls saved"))
                elif p == "Vectorizer":
                    vecs = info.get("loops_vectorized", 0)
                    lines.append(box_line(f"   {p:18s} {vecs} vector loops created"))
            else:
                lines.append(box_line(f"   {p:18s} {info}"))

        lines.append("╚" + "═" * inner_w + "╝")
        return "\n".join(lines)

    def format_markdown_report(self) -> str:
        """Format metrics as a Markdown document with tables."""
        md_lines = [
            f"# CVM Optimization Benchmark Report: `{os.path.basename(self.program)}`",
            "",
            f"- **Program**: `{self.program}`",
            f"- **Timestamp**: `{self.timestamp}`",
            f"- **Correctness**: `{'PASSED (Outputs Match)' if self.correctness else 'FAILED'}`",
            f"- **AI Selected Passes**: `{'`, `'.join(self.selected_passes)}`",
            "",
            "## Summary of Speedup",
            "",
            "| Metric | Ratio / Speedup |",
            "|---|---|",
            f"| **Dynamic Execution Speedup** | **{self.dynamic_speedup:.2f}×** |",
            f"| **Wall Clock Speedup (median)** | **{self.wall_speedup:.2f}×** |",
            f"| **Static Code Size Reduction** | **{self.code_size_reduction_percent:.1f}%** ({self.static_speedup:.2f}×) |",
            "",
            "## Static Analysis",
            "",
            "| Metric | Baseline | Optimized | Change |",
            "|---|---|---|---|",
            f"| IR Statements | {self.instruction_count_before} | {self.instruction_count_after} | {self.code_size_reduction_percent:+.1f}% |",
            f"| Basic Blocks | {self.basic_block_count_before} | {self.basic_block_count_after} | {self.basic_block_count_after - self.basic_block_count_before:+d} |",
            f"| Natural Loops | {self.loop_count_before} | {self.loop_count_after} | {self.loop_count_after - self.loop_count_before:+d} |",
            f"| Dead Code Removed | - | {self.dead_code_removed} | - |",
            f"| Constants Folded | - | {self.constants_folded} | - |",
            f"| Copies Eliminated | - | {self.copies_eliminated} | - |",
            f"| CSE Replacements | - | {self.cse_replacements} | - |",
            f"| LICM Hoisted Instrs | - | {self.instructions_hoisted_by_LICM} | - |",
            f"| Loops Vectorized | - | {self.loops_vectorized} | - |",
            f"| Register Spills | - | {self.estimated_register_spills} | - |",
            "",
            "## Dynamic Execution",
            "",
            "| Metric | Baseline | Optimized | Ratio / Change |",
            "|---|---|---|---|",
            f"| VM Instructions Executed | {self.vm_instructions_executed_before:,} | {self.vm_instructions_executed_after:,} | **{self.dynamic_speedup:.2f}×** faster |",
            f"| Median Wall Time (5 runs) | {self.execution_time_ms_before:.2f} ms | {self.execution_time_ms_after:.2f} ms | **{self.wall_speedup:.2f}×** faster |",
            f"| Peak Operand Stack Depth | {self.operand_stack_peak_depth_before} | {self.operand_stack_peak_depth_after} | {self.operand_stack_peak_depth_after - self.operand_stack_peak_depth_before:+d} |",
            f"| Memory Accesses (Cache Proxy) | {self.estimated_cache_misses_before:,} | {self.estimated_cache_misses_after:,} | {self.estimated_cache_misses_after - self.estimated_cache_misses_before:+d} |",
            f"| Function Calls | {self.function_call_count_before} | {self.function_call_count_after} | {self.function_call_count_after - self.function_call_count_before:+d} |",
            "",
            "## Pass-by-Pass Contribution",
            "",
            "| Pass | Contribution / Gain |",
            "|---|---|",
        ]

        total_gain = self.instruction_count_before - self.instruction_count_after
        for p, info in self.pass_contributions.items():
            if isinstance(info, dict):
                instrs = info.get("instructions_saved", 0)
                if instrs > 0:
                    if total_gain > 0:
                        pct = (instrs / total_gain) * 100.0
                        md_lines.append(f"| `{p}` | −{instrs} instructions ({pct:.1f}% of static gain) |")
                    else:
                        md_lines.append(f"| `{p}` | −{instrs} instructions |")
                elif p == "RegisterAllocation":
                    spills = info.get("spills", 0)
                    regs = info.get("regs_allocated", 0)
                    md_lines.append(f"| `{p}` | {spills} spills, {regs} virtual registers allocated |")
                elif p == "InstructionScheduler":
                    stalls = info.get("stalls_saved", 0)
                    md_lines.append(f"| `{p}` | {stalls} pipeline stalls hidden via scheduling |")
                elif p == "Vectorizer":
                    vecs = info.get("loops_vectorized", 0)
                    md_lines.append(f"| `{p}` | {vecs} loops transformed into 4-wide SIMD |")
            else:
                md_lines.append(f"| `{p}` | {info} |")

        return "\n".join(md_lines) + "\n"


class BenchmarkRunner:
    """Orchestrates dual-compilation, execution profiling, correctness verification, and report generation."""

    def __init__(
        self,
        ir_path: str,
        reports_dir: str = "reports",
        timer_runs: int = 5,
        verbose: bool = True,
    ):
        self.ir_path = ir_path
        self.reports_dir = reports_dir
        self.timer_runs = timer_runs
        self.verbose = verbose

    def _count_basic_blocks(self, functions: List[ir.Function]) -> int:
        """Count total basic blocks across all functions in the IR."""
        total = 0
        for fn in functions:
            try:
                cfg = ir.build_cfg(fn)
                total += len(cfg)
            except Exception:
                pass
        return total

    def _count_natural_loops(self, functions: List[ir.Function]) -> int:
        """Count total natural loops detected across all functions."""
        total = 0
        for fn in functions:
            try:
                cfg = ir.build_cfg(fn)
                n_loops = loops.find_natural_loops(cfg)
                total += len(n_loops)
            except Exception:
                pass
        return total

    def _measure_wall_time(self, code: List[Any], runs: int = 5) -> float:
        """Measure median wall-clock execution time over multiple runs in milliseconds."""
        times = []
        # Warm-up run
        try:
            vm_warmup = VM(code, out=lambda v: None)
            vm_warmup.run()
        except Exception:
            pass

        for _ in range(max(1, runs)):
            vm_inst = VM(code, out=lambda v: None)
            t0 = time.perf_counter()
            vm_inst.run()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        return max(statistics.median(times), 0.001)

    def run(self) -> BenchmarkMetrics:
        """Execute baseline vs. optimized dual-compilation benchmark."""
        if not os.path.exists(self.ir_path):
            raise FileNotFoundError(f"IR file not found: {self.ir_path}")

        with open(self.ir_path, "r", encoding="utf-8") as f:
            source = f.read()

        # -------------------------------------------------------------
        # 1. RUN 1 (BASELINE)
        # -------------------------------------------------------------
        # Parse IR -> direct codegen (no optimizations, no regalloc)
        fns_base = ir.parse(source)
        instr_count_before = sum(len(fn.body) for fn in fns_base)
        bb_count_before = self._count_basic_blocks(fns_base)
        loop_count_before = self._count_natural_loops(fns_base)

        prog_base = codegen.generate(fns_base)
        code_base = prog_base.code

        # Run Verifier on baseline bytecode
        has_vec_base = any(
            ins.op in (
                Op.LOAD_INDEXED, Op.STORE_INDEXED, Op.NEW_ARRAY,
                Op.VLOAD, Op.VSTORE, Op.VADD, Op.VSUB, Op.VMUL, Op.VDIV,
            )
            for ins in code_base
        )
        if not has_vec_base:
            v_res_base = Verifier().verify(code_base)
            if not v_res_base.passed:
                raise verifier.VerificationError("Baseline bytecode verification failed")

        # Execute baseline VM with profiling
        vm_base = ProfilingVM(code_base)
        vm_base.run()
        time_ms_before = self._measure_wall_time(code_base, runs=self.timer_runs)

        # -------------------------------------------------------------
        # 2. RUN 2 (OPTIMIZED)
        # -------------------------------------------------------------
        fns_opt = ir.parse(source)
        ai_res = ai_selector.ai_select_passes(fns_opt)
        selected_passes = list(ai_res.selected_passes)

        fns_opt, reg_map, reg_stats, sched_stats, vec_stats, opt_stats = ai_selector.run_ai_pipeline(
            fns_opt, selected_passes
        )

        instr_count_after = sum(len(fn.body) for fn in fns_opt)
        bb_count_after = self._count_basic_blocks(fns_opt)
        loop_count_after = self._count_natural_loops(fns_opt)

        prog_opt = codegen.generate(fns_opt, reg_map=reg_map)
        code_opt = prog_opt.code

        # Run Verifier on optimized bytecode if pure scalar stack
        has_vec_opt = any(
            ins.op in (
                Op.LOAD_INDEXED, Op.STORE_INDEXED, Op.NEW_ARRAY,
                Op.VLOAD, Op.VSTORE, Op.VADD, Op.VSUB, Op.VMUL, Op.VDIV,
            )
            for ins in code_opt
        )
        if reg_map is None and not has_vec_opt:
            v_res_opt = Verifier().verify(code_opt)
            if not v_res_opt.passed:
                raise verifier.VerificationError("Optimized bytecode verification failed")

        # Execute optimized VM with profiling
        vm_opt = ProfilingVM(code_opt)
        vm_opt.run()
        time_ms_after = self._measure_wall_time(code_opt, runs=self.timer_runs)

        # -------------------------------------------------------------
        # 3. CORRECTNESS VERIFICATION (Mandatory Step)
        # -------------------------------------------------------------
        if vm_base.recorded_output != vm_opt.recorded_output:
            raise CorrectnessError(
                f"CORRECTNESS FAILURE in {self.ir_path}:\n"
                f"Baseline Output:  {vm_base.recorded_output}\n"
                f"Optimized Output: {vm_opt.recorded_output}\n"
                f"Aborting benchmark. Never report speedup for incorrect code!"
            )

        # -------------------------------------------------------------
        # 4. COMPUTE METRICS & PASS CONTRIBUTIONS
        # -------------------------------------------------------------
        dead_removed = opt_stats.dead_instrs_removed if opt_stats else 0
        const_folded = (opt_stats.constants_folded + opt_stats.algebraic_simplified) if opt_stats else 0
        copies_elim = opt_stats.copies_propagated if opt_stats else 0
        cse_repl = opt_stats.expressions_eliminated if opt_stats else 0
        hoisted_licm = opt_stats.instructions_hoisted if opt_stats else 0
        loops_vec = vec_stats.loops_vectorized if vec_stats else 0
        spills = reg_stats.spills if reg_stats else 0

        # Pass contributions dictionary
        pass_contribs: Dict[str, Any] = {}
        if "ConstantFolding" in selected_passes:
            pass_contribs["ConstantFolding"] = {"instructions_saved": const_folded}
        if "CopyPropagation" in selected_passes:
            pass_contribs["CopyPropagation"] = {"instructions_saved": copies_elim}
        if "CSE" in selected_passes:
            pass_contribs["CSE"] = {"instructions_saved": cse_repl}
        if "DeadCodeElimination" in selected_passes:
            pass_contribs["DeadCodeElimination"] = {"instructions_saved": dead_removed}
        if "LICM" in selected_passes:
            pass_contribs["LICM"] = {"instructions_saved": hoisted_licm}
        if "LoopUnrolling" in selected_passes and opt_stats:
            pass_contribs["LoopUnrolling"] = {"loops_unrolled": opt_stats.loops_unrolled}
        if "StrengthReduction" in selected_passes and opt_stats:
            pass_contribs["StrengthReduction"] = {"instructions_saved": opt_stats.operations_reduced}
        if "Vectorizer" in selected_passes:
            pass_contribs["Vectorizer"] = {"loops_vectorized": loops_vec}
        if "RegisterAllocation" in selected_passes:
            regs_used = len(set(v for rmap in (reg_map or {}).values() for v in rmap.values())) if reg_map else 0
            pass_contribs["RegisterAllocation"] = {
                "spills": spills,
                "regs_allocated": regs_used,
            }
        if "InstructionScheduler" in selected_passes and sched_stats:
            pass_contribs["InstructionScheduler"] = {
                "stalls_saved": sched_stats.total_stalls_saved,
            }

        # Derived speedups
        static_speedup = max(1.0, instr_count_before / max(instr_count_after, 1))
        dynamic_speedup = vm_base.steps / max(vm_opt.steps, 1)
        wall_speedup = max(0.01, time_ms_before / max(time_ms_after, 0.001))
        code_size_reduction = (1.0 - instr_count_after / max(instr_count_before, 1)) * 100.0

        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")

        metrics = BenchmarkMetrics(
            program=self.ir_path,
            timestamp=ts_str,
            correctness=True,
            baseline_output=vm_base.recorded_output,
            optimized_output=vm_opt.recorded_output,
            selected_passes=selected_passes,
            instruction_count_before=instr_count_before,
            instruction_count_after=instr_count_after,
            basic_block_count_before=bb_count_before,
            basic_block_count_after=bb_count_after,
            loop_count_before=loop_count_before,
            loop_count_after=loop_count_after,
            dead_code_removed=dead_removed,
            constants_folded=const_folded,
            copies_eliminated=copies_elim,
            cse_replacements=cse_repl,
            loops_vectorized=loops_vec,
            instructions_hoisted_by_LICM=hoisted_licm,
            estimated_register_spills=spills,
            vm_instructions_executed_before=vm_base.steps,
            vm_instructions_executed_after=vm_opt.steps,
            execution_time_ms_before=time_ms_before,
            execution_time_ms_after=time_ms_after,
            operand_stack_peak_depth_before=vm_base.peak_stack_depth,
            operand_stack_peak_depth_after=vm_opt.peak_stack_depth,
            estimated_cache_misses_before=vm_base.memory_accesses,
            estimated_cache_misses_after=vm_opt.memory_accesses,
            function_call_count_before=vm_base.call_count,
            function_call_count_after=vm_opt.call_count,
            static_speedup=static_speedup,
            dynamic_speedup=dynamic_speedup,
            wall_speedup=wall_speedup,
            code_size_reduction_percent=code_size_reduction,
            pass_contributions=pass_contribs,
        )

        # -------------------------------------------------------------
        # 5. WRITE REPORTS
        # -------------------------------------------------------------
        os.makedirs(self.reports_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.ir_path))[0]
        json_path = os.path.join(self.reports_dir, f"{base_name}_{ts_file}.json")
        md_path = os.path.join(self.reports_dir, f"{base_name}_{ts_file}.md")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(metrics.format_markdown_report())

        if self.verbose:
            print(metrics.format_text_report())
            print(f"\n[reports] Saved JSON to: {json_path}")
            print(f"[reports] Saved Markdown to: {md_path}")

        return metrics


def run_regression(directory: str = "examples", verbose: bool = True) -> int:
    """Run regression benchmark across all .ir files in directory.

    Asserts output equivalence and non-regression (dynamic speedup >= 0.95).
    Exits with 0 on total success, 1 on any regression or failure.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.", file=sys.stderr)
        return 1

    ir_files = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".ir") and not f.startswith("divzero")  # divzero tests runtime error
    ])

    if not ir_files:
        print(f"No .ir benchmark programs found in '{directory}'.", file=sys.stderr)
        return 1

    print("=" * 106)
    print(f" CVM BENCHMARK REGRESSION SUITE ({len(ir_files)} programs in '{directory}')")
    print("=" * 106)
    print(
        f"{'Program':<20} | {'Correctness':<11} | {'Base Steps':>10} | {'Opt Steps':>10} | {'Dynamic':>9} | {'Wall':>8} | {'Status':<12}"
    )
    print("-" * 106)

    passed_count = 0
    failed_count = 0
    regression_errors: List[str] = []

    for ir_file in ir_files:
        base_name = os.path.basename(ir_file)
        try:
            runner = BenchmarkRunner(ir_file, timer_runs=3, verbose=False)
            metrics = runner.run()

            # Check for performance regression: dynamic speedup should be >= 0.95
            is_slower = metrics.dynamic_speedup < 0.95
            status = "REGRESSION" if is_slower else "OK"

            if is_slower:
                failed_count += 1
                regression_errors.append(
                    f"{base_name}: Slower performance (dynamic speedup {metrics.dynamic_speedup:.2f}× < 0.95×)"
                )
            else:
                passed_count += 1

            print(
                f"{base_name:<20} | "
                f"{'PASS':<11} | "
                f"{metrics.vm_instructions_executed_before:>10,d} | "
                f"{metrics.vm_instructions_executed_after:>10,d} | "
                f"{metrics.dynamic_speedup:>8.2f}× | "
                f"{metrics.wall_speedup:>7.2f}× | "
                f"{status:<12}"
            )

        except CorrectnessError as e:
            failed_count += 1
            regression_errors.append(f"{base_name}: Correctness Failure ({e})")
            print(
                f"{base_name:<20} | "
                f"{'FAIL':<11} | "
                f"{'-':>10} | "
                f"{'-':>10} | "
                f"{'-':>9} | "
                f"{'-':>8} | "
                f"{'FAIL (output)':<12}"
            )
        except Exception as e:
            failed_count += 1
            regression_errors.append(f"{base_name}: Crash/Error ({e})")
            print(
                f"{base_name:<20} | "
                f"{'ERROR':<11} | "
                f"{'-':>10} | "
                f"{'-':>10} | "
                f"{'-':>9} | "
                f"{'-':>8} | "
                f"{'ERROR':<12}"
            )

    print("=" * 106)
    print(f"Summary: {passed_count}/{len(ir_files)} benchmarks passed.")

    if regression_errors:
        print("\nRegressions / Failures detected:")
        for err in regression_errors:
            print(f"  • {err}")
        return 1

    print("All benchmarks passed correctness and performance regression tests successfully!\n")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for the CVM Benchmark Runner."""
    parser = argparse.ArgumentParser(description="CVM Optimization Benchmark Runner")
    parser.add_argument("target", nargs="?", help="Path to .ir file or directory to benchmark")
    parser.add_argument(
        "--regression",
        metavar="DIR",
        help="Run regression suite over all .ir programs in directory",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of timer runs to calculate median wall-clock time (default: 5)",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory where JSON and Markdown reports are saved (default: reports)",
    )
    args = parser.parse_args()

    if args.regression:
        sys.exit(run_regression(args.regression))
    elif args.target:
        if os.path.isdir(args.target):
            sys.exit(run_regression(args.target))
        else:
            try:
                runner = BenchmarkRunner(
                    args.target,
                    reports_dir=args.reports_dir,
                    timer_runs=args.runs,
                    verbose=True,
                )
                runner.run()
                sys.exit(0)
            except CorrectnessError as e:
                print(f"\n[CORRECTNESS FAILURE] {e}", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"\n[BENCHMARK ERROR] {e}", file=sys.stderr)
                sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
