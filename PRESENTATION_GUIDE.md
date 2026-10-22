# Presentation guide

## Main message

This project compiles a documented integer-focused subset of Python through an AST frontend, three-address intermediate representation, CFG-aware constant propagation and liveness, bytecode generation, and a custom stack virtual machine. It supports booleans as integer truth values and fixed-size tuples with static indexing. Runtime parallelism supports 15 integer reductions. It does not claim to compile arbitrary Python or to make every parallel workload faster.

## Suggested 7-minute presentation

### 1. Problem and scope — 40 seconds

“The goal was to build and demonstrate compiler stages, not to reimplement all of Python. The frontend accepts a defined integer-oriented subset and reports unsupported syntax with a diagnostic.”

Point to the supported-subset slide. Name examples: integer arithmetic, booleans, fixed tuples with constant indexes, branches, loops, functions, recursion, and integer output. Clarify that lists and dynamic indexing remain unsupported.

### 2. Compiler pipeline — 60 seconds

“Python’s AST parser gives us structured syntax. The frontend lowers that syntax into the project’s three-address IR. The optimizer can transform that IR, code generation emits bytecode, and the VM executes it.”

Run the Python sum example with its intermediate stages visible:

```sh
python3 main.py examples/python_sum.py --ir --cfg --dis --stats
```

If asked about functions or recursion, `python3 main.py examples/python_factorial.py --stats` prints `120` and `3628800`.

### 3. IR and control flow — 70 seconds

“A loop is represented with basic blocks and jumps. The CFG makes predecessor and successor relationships explicit. This matters because a value is safe to propagate at a join only when all incoming paths agree.”

Run:

```sh
python3 main.py examples/sum.ir --opt --demo --stats
```

Point out that the loop comparison uses the invariant bound `10`, while the changing loop counter remains a variable.

To show a branch join where both paths agree on the same constant, run `python3 main.py examples/branch_join.ir --opt --ir --cfg --stats`; the optimized output remains `7`.

### 4. Optimization evidence — 60 seconds

"The comparison compiles the same source in two ways, runs both versions, checks their outputs, and reports instruction counts. This 20-iteration scoring loop prints `17220` both ways; executed instructions fall from 693 to 477, and static bytecode falls from 60 to 34."

Run:

```sh
python3 main.py examples/python_optimization_workload.py --compare
```

Then, if time permits, run the loop comparison:

```sh
python3 main.py examples/sum.ir --compare
```

Both versions print `55`; the current loop example changes from 28 to 26 static bytecode instructions and from 169 to 167 executed VM instructions. Explain that improvements depend on the program.

### 5. Runtime parallelism — 90 seconds

“All 15 `parallel_*` reductions send independent integer inputs to isolated worker VMs in separate processes, then combine their results. This example computes a sum of squares, while a second example exercises every reduction.”

Run these commands:

```sh
python3 main.py examples/python_sequential_sum.py --stats
python3 main.py examples/python_parallel_sum.py --stats
```

The current examples both output `332833500`. The sequential workload executes 19,019 VM instructions. The parallel example reports 13 parent instructions and 10,000 worker instructions in the measured environment. The broader reduction example prints a result for each operation. These counts show work performed by worker VMs; they do not alone prove lower elapsed time.

### 6. Timing and limitations — 60 seconds

“Instruction counts and elapsed time answer different questions. The performance study repeats sequential and process-parallel runs and reports timing statistics. Process startup and dispatch are included, so a small parallel workload can lose time even when its worker tasks are independent.”

Run:

```sh
python3 performance_study.py --sizes 1000 5000 --runs 5
```

Only report the timing measured on the presentation machine, and call it a local result. Do not claim a general speedup. Describe unsupported Python as a deliberate boundary of this project.

### 7. Close — 20 seconds

“The project demonstrates a complete path from a documented Python subset to custom bytecode and execution. It includes CFG liveness, fixed tuples, 15 safe integer reductions, repeated performance studies, and versioned bytecode. The remaining work is broader Python semantics and more advanced compiler analyses.”

## Guided demo

From the repository root, run:

```sh
python3 presentation_demo.py
```

The script pauses between steps so you can explain each one. For an unattended run, use `python3 presentation_demo.py --no-pause`. To skip the timing step, add `--skip-benchmark`.

Recommended live order:

1. Python source lowered into IR, CFG, and bytecode.
2. Optimized IR, CFG, bytecode, and output for the loop.
3. Branch-join constant proof, if time permits.
4. Optimized/unoptimized comparison.
5. Sequential reference and process-parallel run.
6. Bounded boolean/tuple examples, then the full reduction example.
7. A deliberate unsupported-feature diagnostic.
8. Repeated sequential-versus-parallel performance study.

## Likely professor questions

**Does this compile all Python?**  
No. It supports the documented integer-oriented subset and rejects unsupported constructs. Full Python requires dynamic object behavior and a much larger runtime.

**What does the CFG optimizer do?**  
It solves constant facts over basic-block edges to a fixed point. At a branch join it retains only constants shared by all incoming paths. The loop back-edge participates, allowing invariant values to propagate while the changing loop counter remains unknown.

**What does liveness do?**  
It analyzes variables backward across CFG edges to identify values that can still be read. The optimizer removes dead pure assignments, while preserving potentially trapping division and modulo operations.

**How do you know optimization preserves behavior?**  
The `--compare` command runs both versions and checks output equality. The test suite also compares outputs across the demonstration programs and directly tests branch joins and loop invariants.

**What exactly is parallel?**  
The `parallel_*` intrinsics evaluate a pure one-argument worker for independent integer inputs in separate processes. The project supports 15 reductions. `--parallel-opt` is a separate compile-time feature that concurrently optimizes independent functions.

**Why restrict the worker?**  
Tasks should not rely on shared mutable state or side effects. The current worker contract allows local integer computations and control flow; it rejects printing, calls, and nested reductions.

**Does parallelism make it faster?**  
Not always. Worker instruction counts demonstrate execution, but process creation and scheduling add overhead. The benchmark measures elapsed time for the selected workload; the result is specific to that machine and run conditions.

**Does the project support Python containers?**  
It supports fixed-size integer tuples assigned to local names with constant indexes. Dynamic indexing, lists, dictionaries, and mutation remain outside the current subset.

**How do the reductions handle special cases?**  
`parallel_count` counts nonzero returns; `parallel_average` uses floor division; argmin/argmax return the input whose result is extreme and break ties by choosing the lowest input. Empty-range identities and errors are defined in the project explanation.

## Before presenting

- Run `python3 tests.py` and confirm it reports 46 passing tests.
- Run `python3 presentation_demo.py --no-pause` once to check the example sequence.
- Run `python3 main.py examples/python_optimization_workload.py --compare` and `python3 main.py examples/sum.ir --compare`; verify outputs are equivalent.
- Confirm sequential and parallel examples both output `332833500`.
- Run `python3 main.py examples/python_types.py --opt --stats` and `python3 main.py examples/python_parallel_reductions.py --stats`.
- Run `python3 performance_study.py --sizes 1000 5000 --runs 5` on the presentation machine and report its median results as local measurements.
- Keep the repository folder available. If a live process-pool example is unreliable in the presentation environment, show the recorded instruction/worker statistics and explain what the automated test verifies.
- Be ready to state the subset boundary and worker restrictions directly.

## Current limitations and next steps

- The frontend does not support arbitrary Python. It supports fixed integer tuples with constant indexing, but not lists, dictionaries, dynamic indexing, mutable containers, classes, imports, exceptions, floats, or the dynamic object model.
- Runtime parallelism supports fifteen deterministic integer reductions over independent pure workers, not general Python multiprocessing. Count, average, argmin, and argmax have specific documented semantics.
- The CFG optimizer includes constant propagation and liveness-based dead assignment removal; SSA and other dataflow passes remain future work.
- `performance_study.py` repeats sequential and process-parallel workloads and checks their outputs. Timings include process startup and are local to the machine and load.
- Bytecode has an explicit version and reads the earlier `CVM1` layout. The decoder rejects unknown future versions; older VMs cannot execute newer opcodes.
