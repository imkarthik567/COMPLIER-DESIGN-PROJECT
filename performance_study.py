"""Repeated sequential-versus-process-parallel workload study."""
import argparse
import statistics
import time

from codegen import generate
from python_frontend import parse_python
from vm import VM


def programs(size):
    sequential = f"""total = 0
for i in range({size}):
    total += i * i
print(total)
"""
    parallel = f"""def square(i):
    return i * i
print(parallel_sum(square, 0, {size}))
"""
    return (generate(parse_python(sequential)), generate(parse_python(parallel)))


def run_once(program):
    vm = VM(program.code, out=lambda _value: None)
    start = time.perf_counter()
    output = vm.run()
    return time.perf_counter() - start, output, vm.steps + vm.parallel_worker_steps, vm.parallel_workers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 5000], help="positive input sizes")
    parser.add_argument("--runs", type=int, default=5, help="repetitions per version and size")
    args = parser.parse_args()
    if args.runs < 2 or any(size < 1 or size > 100_000 for size in args.sizes):
        parser.error("use at least 2 runs and sizes from 1 through 100000")

    print("size | implementation | runs | result | VM steps | median seconds | min-max seconds | workers")
    for size in args.sizes:
        sequential, parallel = programs(size)
        records = {"sequential": [], "parallel": []}
        # Alternate order to reduce consistent warm-up/order bias.
        for run in range(args.runs):
            order = (("sequential", sequential), ("parallel", parallel))
            if run % 2:
                order = tuple(reversed(order))
            for label, program in order:
                records[label].append(run_once(program))
        outputs = {name: rows[0][1] for name, rows in records.items()}
        if outputs["sequential"] != outputs["parallel"]:
            raise RuntimeError(f"result mismatch at size {size}: {outputs}")
        for label in ("sequential", "parallel"):
            rows = records[label]
            times = [row[0] for row in rows]
            print(f"{size} | {label} | {len(rows)} | {outputs[label][0]} | {rows[0][2]} | "
                  f"{statistics.median(times):.6f} | {min(times):.6f}-{max(times):.6f} | {rows[0][3]}")
        seq = statistics.median(r[0] for r in records["sequential"])
        par = statistics.median(r[0] for r in records["parallel"])
        ratio = seq / par if par else float("inf")
        print(f"  same result: yes; sequential/parallel median-time ratio: {ratio:.3f}x "
              "(greater than 1 means parallel was faster on this run)\n")
    print("Timings include VM setup and process-pool startup; they describe this machine and run only.")


if __name__ == "__main__":
    main()
