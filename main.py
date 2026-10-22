import argparse
import statistics
import sys
import time

from codegen import CodegenError, generate
from ir import IRError, format_cfg, format_ir, parse
from isa import decode, disassemble, encode
from vm import VM, VMError
from optimizer import optimize
from python_frontend import parse_python


def banner(title):
    print(f"\n=== {title} " + "=" * max(3, 60 - len(title)))


def main():
    ap = argparse.ArgumentParser(description="CVM compiler backend + virtual machine")
    ap.add_argument("file", help=".ir source or .bc bytecode file")
    ap.add_argument("--demo", action="store_true", help="show IR, CFG and bytecode, then run")
    ap.add_argument("--ir", action="store_true", help="print the parsed IR")
    ap.add_argument("--cfg", action="store_true", help="print basic blocks and CFG edges")
    ap.add_argument("--dis", action="store_true", help="print bytecode disassembly")
    ap.add_argument("--trace", action="store_true", help="trace every VM instruction")
    ap.add_argument("--opt", action="store_true", help="optimize IR before code generation")
    ap.add_argument("--parallel-opt", "--parallel", dest="parallel_opt", action="store_true",
                    help="optimize independent functions concurrently (implies --opt)")
    ap.add_argument("--stats", action="store_true", help="show static and executed instruction counts")
    ap.add_argument("--compare", action="store_true", help="compile and run baseline and optimized versions, then compare them")
    ap.add_argument("--benchmark", type=int, metavar="RUNS",
                    help="run baseline and optimized versions repeatedly and report wall-clock timings")
    ap.add_argument("-o", "--emit", metavar="OUT.bc", help="write binary bytecode to a file")
    ap.add_argument("--no-run", action="store_true", help="compile only")
    args = ap.parse_args()

    if args.demo:
        args.ir = args.cfg = args.dis = True

    try:
        if args.file.endswith(".bc"):
            if args.compare or args.benchmark is not None:
                raise ValueError("--compare/--benchmark require Python source (.py) or three-address IR (.ir), not bytecode")
            with open(args.file, "rb") as f:
                code = decode(f.read())
            funcs = labels = None
            print(f"loaded {len(code)} instructions from {args.file}")
        else:
            with open(args.file) as f:
                source = f.read()
            functions = parse_python(source) if args.file.endswith(".py") else parse(source)
            original_count = sum(len(fn.body) for fn in functions)
            if args.benchmark is not None and args.benchmark < 1:
                raise ValueError("--benchmark RUNS must be at least 1")
            if args.compare and args.benchmark is not None:
                raise ValueError("choose either --compare or --benchmark")
            if args.compare:
                baseline = generate(functions)
                optimized_functions = optimize(functions, parallel=args.parallel_opt)
                optimized = generate(optimized_functions)
                baseline_vm = VM(baseline.code, out=lambda _value: None)
                optimized_vm = VM(optimized.code, out=lambda _value: None)
                baseline_output = baseline_vm.run()
                optimized_output = optimized_vm.run()
                for label, vm, program, output, ir_count in (
                    ("baseline", baseline_vm, baseline, baseline_output, original_count),
                    ("optimized", optimized_vm, optimized, optimized_output,
                     sum(len(fn.body) for fn in optimized_functions)),
                ):
                    print(f"[compare] {label}")
                    print(f"  output: {output}")
                    print(f"  IR statements: {ir_count}")
                    print(f"  static bytecode instructions: {len(program.code)}")
                    print(f"  parent VM instructions: {vm.steps}")
                    total_steps = vm.steps + vm.parallel_worker_steps
                    if vm.parallel_tasks:
                        print(f"  parallel worker VM instructions: {vm.parallel_worker_steps}")
                        print(f"  parallel tasks / max workers: {vm.parallel_tasks} / {vm.parallel_workers}")
                    print(f"  counted VM instructions including workers: {total_steps}")
                print(f"[compare] outputs equivalent: {'yes' if baseline_output == optimized_output else 'NO'}")
                if baseline_vm.steps:
                    baseline_steps = baseline_vm.steps + baseline_vm.parallel_worker_steps
                    optimized_steps = optimized_vm.steps + optimized_vm.parallel_worker_steps
                    reduction = 100 * (baseline_steps - optimized_steps) / baseline_steps
                    direction = "fewer" if reduction >= 0 else "more"
                    print(f"[compare] counted VM instruction change: {abs(reduction):.2f}% {direction}")
                    static_delta = 100 * (len(baseline.code) - len(optimized.code)) / len(baseline.code)
                    direction = "fewer" if static_delta >= 0 else "more"
                    print(f"[compare] static bytecode change: {abs(static_delta):.2f}% {direction}")
                return 0 if baseline_output == optimized_output else 1
            if args.benchmark is not None:
                baseline = generate(functions)
                optimized_functions = optimize(functions, parallel=args.parallel_opt)
                optimized = generate(optimized_functions)
                results = []
                for label, program in (("baseline", baseline), ("optimized", optimized)):
                    samples, output, vm = [], None, None
                    for _ in range(args.benchmark):
                        vm = VM(program.code, out=lambda _value: None)
                        start = time.perf_counter()
                        current_output = vm.run()
                        samples.append(time.perf_counter() - start)
                        if output is not None and current_output != output:
                            raise VMError(f"{label} output changed between benchmark runs")
                        output = current_output
                    results.append((label, program, samples, output, vm))
                equivalent = results[0][3] == results[1][3]
                for label, program, samples, output, vm in results:
                    print(f"[benchmark] {label} ({len(samples)} run(s))")
                    print(f"  output: {output}")
                    print(f"  static bytecode instructions: {len(program.code)}")
                    print(f"  VM instructions per run: {vm.steps}")
                    if vm.parallel_tasks:
                        print(f"  worker VM instructions per run: {vm.parallel_worker_steps}")
                        print(f"  parallel tasks / max workers: {vm.parallel_tasks} / {vm.parallel_workers}")
                    print(f"  wall time min / median / mean: {min(samples):.6f} / {statistics.median(samples):.6f} / {statistics.mean(samples):.6f} seconds")
                print(f"[benchmark] outputs equivalent: {'yes' if equivalent else 'NO'}")
                print("[benchmark] timings include VM setup and process-pool startup where parallel workers are used; results vary by machine and load")
                return 0 if equivalent else 1
            if args.parallel_opt:
                args.opt = True
            if args.opt:
                functions = optimize(functions, parallel=args.parallel_opt)
            if args.ir:
                banner("1. INTERMEDIATE REPRESENTATION (three-address code)")
                print(format_ir(functions))
            if args.cfg:
                banner("2. BASIC BLOCKS AND CONTROL FLOW GRAPH")
                print(format_cfg(functions))
            program = generate(functions)
            code, funcs, labels = program.code, program.funcs, program.labels
            if args.stats:
                print(f"[stats] IR statements: {original_count} -> {sum(len(fn.body) for fn in functions)}")
                print(f"[stats] source frontend: {'Python subset' if args.file.endswith('.py') else 'three-address IR'}")

        if args.dis:
            banner(f"3. BYTECODE ({len(code)} instructions)")
            print(disassemble(code, funcs, labels))

        if args.emit:
            data = encode(code)
            with open(args.emit, "wb") as f:
                f.write(data)
            print(f"\nwrote {len(data)} bytes to {args.emit}")

        if args.no_run:
            return 0

        banner("4. VM EXECUTION" if (args.demo or args.dis) else "OUTPUT")
        vm = VM(code, trace=args.trace)
        vm.run()
        print(f"\n[vm] halted normally after {vm.steps} instructions")
        if args.stats:
            print(f"[stats] executed bytecode instructions: {vm.steps}")
            print(f"[stats] static bytecode instructions: {len(code)}")
            if vm.parallel_tasks:
                print(f"[parallel] processed {vm.parallel_tasks} independent inputs across up to {vm.parallel_workers} worker processes")
                print(f"[parallel] worker VM instructions: {vm.parallel_worker_steps}")
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
