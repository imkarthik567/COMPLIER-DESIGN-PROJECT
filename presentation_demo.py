#!/usr/bin/env python3
"""Run a guided, presentation-ready demonstration of the compiler project."""
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def pause(enabled):
    if not enabled:
        return
    try:
        input("\nPress Enter to continue... ")
    except EOFError:
        pass


def run_step(title, explanation, args, *, expected_failure=False):
    print("\n" + "=" * 76)
    print(title)
    print(explanation)
    print("$ " + " ".join([PYTHON, *args]))
    sys.stdout.flush()
    result = subprocess.run([PYTHON, *args], cwd=ROOT, check=False)
    if expected_failure:
        if result.returncode == 0:
            raise SystemExit("Expected the unsupported-syntax example to be rejected, but it succeeded.")
        print("[demo] expected result: compiler rejected unsupported syntax")
    elif result.returncode != 0:
        raise SystemExit(f"Demo step failed with exit code {result.returncode}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-pause", action="store_true", help="run all steps without waiting for Enter")
    parser.add_argument("--skip-benchmark", action="store_true", help="skip the repeated performance study at the end")
    args = parser.parse_args()
    wait = not args.no_pause

    print("COMPILER DESIGN PROJECT — GUIDED DEMO")
    print("This walkthrough uses examples in the repository and checks each command's result.")
    pause(wait)

    run_step(
        "1. Python source through the compiler pipeline",
        "Inspect Python source after it has become IR, CFG blocks, and bytecode, then run it on the VM.",
        ["main.py", "examples/python_sum.py", "--ir", "--cfg", "--dis", "--stats"],
    )
    pause(wait)

    run_step(
        "2. IR, control-flow graph, bytecode, and execution",
        "The loop example exposes the intermediate stages; optimization propagates the invariant bound into the loop test.",
        ["main.py", "examples/sum.ir", "--opt", "--demo", "--stats"],
    )
    pause(wait)

    run_step(
        "3. Constant agreement at a branch join",
        "Both branches assign 7. The CFG analysis proves that value at the join and the output stays 7.",
        ["main.py", "examples/branch_join.ir", "--opt", "--ir", "--cfg", "--stats"],
    )
    pause(wait)

    run_step(
        "4. Optimized versus unoptimized execution",
        "Both versions compile from the same Python source. The command verifies equal output and reports instruction-count changes.",
        ["main.py", "examples/python_optimization_workload.py", "--compare"],
    )
    pause(wait)

    run_step(
        "5. Sequential reference result",
        "Run the sequential sum-of-squares example and note its output and VM instruction count.",
        ["main.py", "examples/python_sequential_sum.py", "--stats"],
    )
    pause(wait)

    run_step(
        "6. Process-level parallel worker execution",
        "Run the same sum-of-squares task with independent worker inputs. Point out the process and worker instruction statistics.",
        ["main.py", "examples/python_parallel_sum.py", "--stats"],
    )
    pause(wait)

    run_step(
        "7. Boolean and fixed-tuple support",
        "This bounded Python extension demonstrates boolean truth values and a fixed tuple accessed with compile-time indexes.",
        ["main.py", "examples/python_types.py", "--opt", "--stats"],
    )
    pause(wait)

    run_step(
        "8. Full parallel reduction set",
        "The example runs all 15 supported operations, including product, count, logical, bitwise, GCD/LCM, average, and argmin/argmax reductions.",
        ["main.py", "examples/python_parallel_reductions.py", "--stats"],
    )
    pause(wait)

    run_step(
        "9. A deliberate subset boundary",
        "A list example is intentionally outside this compiler's documented Python subset; the frontend reports an error instead of silently compiling it incorrectly.",
        ["main.py", "examples/unsupported_python.py"],
        expected_failure=True,
    )

    if not args.skip_benchmark:
        pause(wait)
        run_step(
            "10. Repeated sequential-versus-parallel study",
            "Three runs at 1,000 inputs check equal results and measure local sequential and process-parallel time. Process overhead can dominate.",
            ["performance_study.py", "--sizes", "1000", "--runs", "3"],
        )

    print("\nDemo complete. For a short presentation, omit steps 7–9 or the performance study if time is limited.")


if __name__ == "__main__":
    main()
