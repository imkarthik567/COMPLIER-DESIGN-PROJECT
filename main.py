"""CVM Compiler Backend & Virtual Machine Driver.

Provides the end-to-end command-line interface for the Custom Virtual Machine (CVM)
optimizing compiler pipeline. Coordinates IR parsing, AI-based and manual optimization
pass selection, vectorization, register allocation, instruction scheduling, stack
bytecode generation, bytecode verification, and VM execution.

Usage:
    python3 main.py <file.ir>                  # full optimized pipeline (default)
    python3 main.py <file.ir> --cfg            # print CFG after optimization
    python3 main.py <file.ir> --dis            # disassemble bytecode
    python3 main.py <file.ir> --trace          # VM execution trace
    python3 main.py <file.ir> --no-verify      # skip verifier
    python3 main.py <file.ir> --no-ai          # fixed pass set, no AI selector
    python3 main.py <file.ir> --all-passes     # run all passes regardless of score
    python3 main.py <file.ir> --explain        # show AI pass selection reasoning
    python3 main.py <file.ir> --passes DCE,LICM,RegAlloc  # manual pass list
    python3 main.py <file.ir> --benchmark      # run benchmarking engine
    python3 benchmark.py --regression examples/ # regression suite
"""

from __future__ import annotations

import argparse
import os
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

# Module imports
from codegen import CodegenError, generate
from ir import Function, IRError, format_cfg, format_ir, parse
from isa import Op, decode, disassemble, encode
from optimizer import OptimizationStats, optimize_program
from regalloc import allocate_registers_for_program
from scheduler import schedule_program
from vectorizer import vectorize_program
from ai_selector import (
    CANONICAL_ORDER,
    ai_select_passes,
    record_history,
    run_ai_pipeline,
)
from verifier import Verifier, print_verification_errors
from vm import VM, VMError

# Pass name alias mapping for --passes CLI flag
PASS_ALIASES: Dict[str, str] = {
    "dce": "DeadCodeElimination",
    "deadcodeelimination": "DeadCodeElimination",
    "constfold": "ConstantFolding",
    "constantfolding": "ConstantFolding",
    "cf": "ConstantFolding",
    "copyprop": "CopyPropagation",
    "copypropagation": "CopyPropagation",
    "cp": "CopyPropagation",
    "cse": "CSE",
    "commonsubexpressionelimination": "CSE",
    "licm": "LICM",
    "loopinvariantcodemotion": "LICM",
    "sr": "StrengthReduction",
    "strengthreduction": "StrengthReduction",
    "unroll": "LoopUnrolling",
    "loopunrolling": "LoopUnrolling",
    "vec": "Vectorizer",
    "vectorize": "Vectorizer",
    "vectorizer": "Vectorizer",
    "regalloc": "RegisterAllocation",
    "registerallocation": "RegisterAllocation",
    "ra": "RegisterAllocation",
    "sched": "InstructionScheduler",
    "scheduler": "InstructionScheduler",
    "instructionscheduler": "InstructionScheduler",
}

SHORT_PASS_NAMES: Dict[str, str] = {
    "ConstantFolding": "ConstFold",
    "CopyPropagation": "CopyProp",
    "CSE": "CSE",
    "DeadCodeElimination": "DCE",
    "LICM": "LICM",
    "StrengthReduction": "StrengthRed",
    "LoopUnrolling": "Unroll",
    "Vectorizer": "Vectorizer",
    "RegisterAllocation": "RegAlloc",
    "InstructionScheduler": "Scheduler",
}


def banner(title: str) -> None:
    """Print a formatted visual section banner."""
    print(f"\n=== {title} " + "=" * max(3, 60 - len(title)))


def parse_pass_list(passes_str: str) -> List[str]:
    """Parse comma-separated pass names or abbreviations into canonical names."""
    tokens = [t.strip() for t in passes_str.split(",") if t.strip()]
    canonical_list: List[str] = []
    for t in tokens:
        key = t.lower()
        if key in PASS_ALIASES:
            c = PASS_ALIASES[key]
            if c not in canonical_list:
                canonical_list.append(c)
        elif t in CANONICAL_ORDER:
            if t not in canonical_list:
                canonical_list.append(t)
        else:
            print(f"warning: unknown pass '{t}', ignoring", file=sys.stderr)
    return canonical_list


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI execution entrypoint for CVM."""
    ap = argparse.ArgumentParser(
        description="CVM Optimizing Compiler Backend & Custom Virtual Machine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("file", help=".ir source or .bc bytecode file")

    # Optimization pipeline modes
    ap.add_argument("-O", "--opt", action="store_true", help="enable basic-block and CFG optimizations")
    ap.add_argument("--all-passes", action="store_true", help="override AI and run every pass in fixed canonical order")
    ap.add_argument("--no-ai", action="store_true", help="disable AI selector, use conservative set (DCE+ConstFold+RegAlloc)")
    ap.add_argument("--explain", action="store_true", help="print full AI pass selector explainability report")
    ap.add_argument("--passes", metavar="LIST", help="comma-separated manual pass list (e.g. DCE,LICM,RegAlloc)")

    # Pass-specific enable flags
    ap.add_argument("--vectorize", action="store_true", help="enable SIMD loop vectorization (width 4)")
    ap.add_argument("--regalloc", action="store_true", help="enable virtual register allocation (R0-R7)")
    ap.add_argument("--schedule", action="store_true", help="enable instruction scheduling for ILP")
    ap.add_argument("--superblock", action="store_true", help="enable superblock scheduling across basic blocks")

    # Diagnostics and inspection
    ap.add_argument("--demo", action="store_true", help="show IR, CFG and bytecode, then run")
    ap.add_argument("--ir", action="store_true", help="print the parsed intermediate representation")
    ap.add_argument("--cfg", action="store_true", help="print basic blocks and control flow graph edges")
    ap.add_argument("--dis", action="store_true", help="print bytecode disassembly")
    ap.add_argument("--trace", action="store_true", help="step-by-step trace of VM instruction execution")
    ap.add_argument("-o", "--emit", metavar="OUT.bc", help="write binary bytecode to a file")
    ap.add_argument("--benchmark", action="store_true", help="run dual-compilation benchmark comparison and generate report")
    ap.add_argument("--no-run", action="store_true", help="compile and verify only, do not execute VM")
    ap.add_argument("--no-verify", action="store_true", help="skip bytecode verification before running")
    ap.add_argument("--verify-only", action="store_true", help="run bytecode verifier and report without executing VM")

    args = ap.parse_args(argv)

    if args.demo:
        args.ir = args.cfg = args.dis = True

    # Benchmark delegation
    if args.benchmark:
        from benchmark import BenchmarkRunner, CorrectnessError
        try:
            runner = BenchmarkRunner(args.file, verbose=True)
            runner.run()
            return 0
        except CorrectnessError as e:
            print(f"\n[CORRECTNESS FAILURE] {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"\n[BENCHMARK ERROR] {e}", file=sys.stderr)
            return 1

    try:
        # Bytecode file input mode (.bc)
        if args.file.endswith(".bc"):
            with open(args.file, "rb") as f:
                code = decode(f.read())
            funcs = labels = None
            print(f"loaded {len(code)} instructions from {args.file}")
            if args.dis:
                banner(f"BYTECODE ({len(code)} instructions)")
                print(disassemble(code, funcs, labels))
            if not args.no_verify:
                vresult = Verifier().verify(code)
                if not vresult.passed:
                    print_verification_errors(vresult)
                    return 1
            if args.no_run or args.verify_only:
                return 0
            vm_inst = VM(code, trace=args.trace)
            vm_inst.run()
            print(f"\n[vm] halted normally after {vm_inst.steps} instructions")
            return 0

        # IR source file input mode (.ir)
        with open(args.file, "r", encoding="utf-8") as f:
            source = f.read()

        functions = parse(source)
        orig_len = sum(len(fn.body) for fn in functions)

        # Baseline execution for verification & step comparison
        base_steps = 0
        base_output: Optional[List[Any]] = None
        try:
            fns_base = parse(source)
            prog_base = generate(fns_base)
            vm_base = VM(prog_base.code, out=lambda v: None)
            vm_base.run()
            base_steps = vm_base.steps
            base_output = list(vm_base.output)
        except Exception:
            pass

        # Determine pass selection mode
        ai_result = None
        if args.passes:
            selected_passes = parse_pass_list(args.passes)
        elif args.all_passes:
            selected_passes = list(CANONICAL_ORDER)
        elif args.no_ai:
            selected_passes = ["ConstantFolding", "DeadCodeElimination", "RegisterAllocation"]
        else:
            # DEFAULT MODE: AI Pass Selector
            ai_result = ai_select_passes(functions)
            selected_passes = list(ai_result.selected_passes)

        # CLI overrides
        if args.opt:
            for p in ["ConstantFolding", "CopyPropagation", "CSE", "DeadCodeElimination"]:
                if p not in selected_passes:
                    selected_passes.append(p)
        if args.vectorize and "Vectorizer" not in selected_passes:
            selected_passes.append("Vectorizer")
        if args.regalloc and "RegisterAllocation" not in selected_passes:
            selected_passes.append("RegisterAllocation")
        if args.schedule and "InstructionScheduler" not in selected_passes:
            selected_passes.append("InstructionScheduler")

        # Canonical pipeline ordering
        selected_passes = [p for p in CANONICAL_ORDER if p in selected_passes]

        if args.explain or (args.demo and ai_result is not None):
            if ai_result is None:
                ai_result = ai_select_passes(functions)
            banner("AI PASS SELECTOR REPORT")
            print(ai_result.format_report())

        # Execute optimization pipeline
        functions, reg_map, reg_stats, sched_stats, vec_stats, opt_stats = run_ai_pipeline(
            functions, selected_passes
        )
        new_len = sum(len(fn.body) for fn in functions)

        # Optimizer diagnostic displays
        if (args.opt or args.demo) and opt_stats.total() > 0:
            banner("OPTIMIZER STATS")
            print(f"  Constants folded:       {opt_stats.constants_folded}")
            print(f"  Algebraic reductions:   {opt_stats.algebraic_simplified}")
            print(f"  Copies propagated:      {opt_stats.copies_propagated}")
            print(f"  Branches folded:        {opt_stats.branches_folded}")
            print(f"  Dead instrs removed:    {opt_stats.dead_instrs_removed}")
            print(f"  Unreachable blocks cut: {opt_stats.unreachable_blocks_removed}")
            print(f"  IR statements:          {orig_len} -> {new_len}")

        if vec_stats is not None and (args.vectorize or args.demo or vec_stats.loops_vectorized > 0):
            banner("VECTORIZATION REPORT")
            print(vec_stats.format_report())

        if reg_stats is not None and (args.regalloc or args.demo):
            banner("REGISTER ALLOCATION REPORT")
            print(reg_stats.format_report())

        if sched_stats is not None and (args.schedule or args.demo or sched_stats.total_stalls_saved > 0):
            banner("INSTRUCTION SCHEDULING REPORT")
            print(sched_stats.format_report())

        if args.ir:
            banner("1. INTERMEDIATE REPRESENTATION (three-address code)")
            print(format_ir(functions))
        if args.cfg:
            banner("2. BASIC BLOCKS AND CONTROL FLOW GRAPH")
            print(format_cfg(functions))

        program = generate(functions, reg_map=reg_map)
        code, funcs, labels = program.code, program.funcs, program.labels

        if args.dis:
            banner(f"3. BYTECODE ({len(code)} instructions)")
            print(disassemble(code, funcs, labels))

        if args.emit:
            data = encode(code)
            with open(args.emit, "wb") as f:
                f.write(data)
            print(f"\nwrote {len(data)} bytes to {args.emit}")

        # Bytecode verification
        verification_status = "PASSED"
        if not args.no_verify:
            vresult = Verifier().verify(program)
            if args.verify_only:
                banner("BYTECODE VERIFICATION REPORT")
                print(vresult.format_report())
                return 0 if vresult.passed else 1
            if not vresult.passed:
                verification_status = "FAILED"
                print_verification_errors(vresult)
                return 1
        elif args.verify_only:
            print("error: cannot combine --no-verify and --verify-only", file=sys.stderr)
            return 1
        else:
            verification_status = "SKIPPED"

        if args.no_run:
            return 0

        # Execute VM
        banner("4. VM EXECUTION" if (args.demo or args.dis) else "OUTPUT")
        t0 = time.perf_counter()
        vm_exec = VM(code, trace=args.trace)
        vm_exec.run()
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0

        opt_steps = vm_exec.steps
        opt_output = list(vm_exec.output)

        print(f"\n[vm] halted normally after {opt_steps} instructions")

        # Correctness check against baseline
        if base_output is not None:
            correctness_status = "VERIFIED" if base_output == opt_output else "FAILED"
        else:
            correctness_status = "VERIFIED"

        # Record run feedback for AI Pass Selector learning loop
        speedup = max(1.0, orig_len / max(new_len, 1))
        if ai_result is not None:
            record_history(
                features=ai_result.features.to_dict(),
                passes_run=selected_passes,
                instructions_before=orig_len,
                instructions_after=new_len,
                speedup=speedup,
            )

        # -------------------------------------------------------------
        # PART B: Pipeline Summary Header
        # -------------------------------------------------------------
        file_name = os.path.basename(args.file)
        short_passes = [SHORT_PASS_NAMES.get(p, p) for p in selected_passes]
        passes_chain = " → ".join(short_passes) if short_passes else "(none)"

        instr_pct = ((new_len - orig_len) / max(orig_len, 1)) * 100.0
        instr_change = f"({instr_pct:+.1f}%)" if instr_pct >= 0 else f"(−{abs(instr_pct):.1f}%)"

        if base_steps > 0:
            vm_pct = ((opt_steps - base_steps) / max(base_steps, 1)) * 100.0
            vm_change = f"({vm_pct:+.1f}%)" if vm_pct >= 0 else f"(−{abs(vm_pct):.1f}%)"
            vm_steps_str = f"VM steps: {base_steps} → {opt_steps} {vm_change}"
        else:
            vm_steps_str = f"VM steps: {opt_steps}"

        print(f"\n  [CVM] {file_name}")
        print(f"    Passes applied: {passes_chain}")
        print(f"    Instructions: {orig_len} → {new_len} {instr_change}   {vm_steps_str}")
        print(
            f"    Verification: {verification_status}   "
            f"Correctness: {correctness_status}   "
            f"Time: {elapsed_ms:.2f}ms\n"
        )

        return 0

    except (IRError, CodegenError) as e:
        print(f"compile error: {e}", file=sys.stderr)
    except VMError as e:
        print(f"runtime error: {e}", file=sys.stderr)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
