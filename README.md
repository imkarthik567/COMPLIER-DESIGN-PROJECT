# Design and Implementation of Compiler Backend with a Custom VM

An optimizing compiler backend and virtual machine environment implemented in Python. CVM transforms high-level three-address code intermediate representation (IR) into verified stack-and-register bytecode through an adaptive optimization pipeline powered by an AI-based pass selector, graph-coloring register allocation, dependency-graph instruction scheduling, and SIMD loop vectorization.

---

## 1. Overview

CVM (Custom Virtual Machine) implements a complete compiler backend pipeline from three-address intermediate representation down to execution on a 32-bit stack-based virtual machine with virtual register enhancements.

Key technical highlights include:
- **Dataflow Framework**: Generic worklist engine for forward/backward dataflow analysis (Liveness, Reaching Definitions, Available Expressions).
- **Loop Optimizations**: Cooper-Harvey-Kennedy dominator tree computation, natural loop detection, Loop-Invariant Code Motion (LICM), strength reduction, and loop unrolling.
- **SIMD Vectorization**: Affine array dependence testing via the Greatest Common Divisor (GCD) formulation and 4-wide vector IR transformation.
- **Register Allocation**: Chaitin-Briggs graph coloring register allocation with George-Briggs conservative coalescing and spill code generation across 8 virtual registers ($R_0$–$R_7$).
- **Instruction Level Parallelism (ILP)**: Directed Data Dependency Graph (DDG) construction and critical-path list scheduling with superblock extension.
- **Bytecode Verifier**: Static pre-execution safety verification analogous to the JVM bytecode verifier (structural validity, operand bounds, type safety, control flow merge consistency).
- **AI-Based Pass Selection**: Feature-based heuristic scoring engine with history feedback learning that dynamically tailors pass sequences to program characteristics.
- **Dual-Compilation Benchmarking**: Automated comparison engine measuring static code reduction, VM execution steps, median wall-clock timing, operand stack depth, and cache proxy metrics.

---

## 2. Architecture Diagram

```
                 +--------------------------------------+
                 |     Three-Address Code IR (.ir)      |
                 +--------------------------------------+
                                    |
                                    v
                 +--------------------------------------+
                 |      IR Parser & CFG Builder         |
                 +--------------------------------------+
                                    |
                                    v
                 +--------------------------------------+
                 |     Program Feature Extractor        |
                 | (structural, expression, variables)  |
                 +--------------------------------------+
                                    |
                                    v
                 +--------------------------------------+
                 |        AI Pass Selector Engine       |
                 | (PassScoringModel + History Feedback)|
                 +--------------------------------------+
                                    |
                                    | Selected Pass Sequence
                                    v
      ====================== OPTIMIZATION PIPELINE ======================
      |  [1] Loop-Invariant Code Motion (LICM)                          |
      |  [2] Induction Variable Strength Reduction                      |
      |  [3] Loop Unrolling (trip-count <= 8)                           |
      |  [4] Constant Folding & Propagation                             |
      |  [5] Copy Propagation                                           |
      |  [6] Common Subexpression Elimination (CSE)                     |
      |  [7] Dead Code Elimination (DCE via Liveness)                   |
      |  [8] SIMD Loop Vectorizer (GCD dependence test, 4-wide)         |
      |  [9] Register Allocation (Chaitin-Briggs Graph Coloring, R0-R7)  |
      | [10] Instruction Scheduling (DDG Critical-Path List Sched)      |
      ===================================================================
                                    |
                                    v
                 +--------------------------------------+
                 |          Bytecode Generator          |
                 | (9-byte fixed binary format / CVM1)  |
                 +--------------------------------------+
                                    |
                                    v
                 +--------------------------------------+
                 |     Static Bytecode Verifier         |
                 | (safety, type consistency, merges)   |
                 +--------------------------------------+
                                    |
                                    v
                 +--------------------------------------+
                 |       Stack Virtual Machine (VM)     |
                 | (operand stack, call frames, tracing)|
                 +--------------------------------------+
```

---

## 3. Pipeline Stages

1. **IR Parsing & CFG Construction (`ir.py`)**: Parses three-address code instructions (`Assign`, `BinOp`, `IfZ`, `IfNz`, `Goto`, `Call`, `Return`, `Print`, `NewArray`, `LoadIndexed`, `StoreIndexed`) into discrete basic blocks with predecessor and successor edges.
2. **Data-Flow Analysis (`dataflow.py`)**: Abstract iterative worklist framework (`DataFlowAnalysis`) running backward liveness, forward reaching definitions, and forward available expressions until fixed-point convergence.
3. **Loop Analysis & Dominator Trees (`loops.py`)**: Computes immediate dominators (`idom`) using the Cooper-Harvey-Kennedy iterative algorithm, identifies back-edges $(A \to B \text{ where } B \text{ dom } A)$, discovers natural loops, and determines primary induction variables.
4. **Code Optimization (`optimizer.py`)**: Executes classical and loop-specific transformations with full statistics tracking.
5. **SIMD Vectorization (`vectorizer.py`)**: Analyzes loop array accesses with the GCD dependence test and converts scalar memory and arithmetic sequences into 4-wide vector operations (`VLoad`, `VStore`, `VAdd`, `VSub`, `VMul`, `VDiv`).
6. **Register Allocation (`regalloc.py`)**: Constructs variable interference graphs, executes George-Briggs conservative coalescing, and colors the graph with $K=8$ physical registers, inserting spill loads/stores when needed.
7. **Instruction Scheduling (`scheduler.py`)**: Constructs Directed Data Dependency Graphs (DDG) mapping RAW, WAR, WAW, and Control dependencies per basic block and reorders instructions using latency-weighted critical-path list scheduling. Superblock scheduling traces high-frequency branch paths.
8. **Bytecode Generation (`codegen.py`)**: Emits binary bytecode (9 bytes per instruction: 1-byte opcode, 4-byte operand `a`, 4-byte operand `b`) preceded by the `"CVM1"` file header.
9. **Bytecode Verification (`verifier.py`)**: Performs abstract type simulation across all control-flow paths, ensuring type safety, operand validity, jump target boundaries, and stack height merge consistency.
10. **VM Execution (`vm.py`)**: Executes bytecode using an operand stack, activation call frames, virtual register storage, and integrated execution tracing.

---

## 4. Optimization Passes

Each pass inherits from `OptimizationPass` and implements `run(cfg, dataflow_results) -> (cfg, stats)`:

| Pass Name | One-Line Academic Description |
|---|---|
| **Constant Folding** | Evaluates constant arithmetic, comparisons, and modulo at compile time and simplifies algebraic identities ($x+0$, $x\times 1$, $x-x$). |
| **Copy Propagation** | Propagates variable copy assignments ($x = y$) into downstream uses, collapsing redundant assignment chains. |
| **Common Subexpression Elimination (CSE)** | Replaces redundant computations of previously computed expressions using forward available expressions analysis. |
| **Dead Code Elimination (DCE)** | Removes variable assignments whose defined target is not live at the assignment point using backward liveness analysis. |
| **Loop-Invariant Code Motion (LICM)** | Hoists loop-invariant computations whose operands are constant or defined outside the loop into a dedicated loop preheader block. |
| **Strength Reduction** | Replaces expensive loop induction variable multiplications ($t = i \times c$) with incremental additions ($t = t + c$) in the loop body. |
| **Loop Unrolling** | Replicates loop bodies with statically known trip counts ($\le 8$), eliminating loop control branch and increment overhead. |
| **SIMD Vectorizer** | Vectorizes counted loops with affine array accesses into 4-wide vector operations using GCD dependence test verification. |
| **Register Allocation** | Maps virtual variable temporaries to $R_0$–$R_7$ via Chaitin-Briggs graph coloring, eliminating operand stack loads/stores. |
| **Instruction Scheduling** | Reorders instructions within basic blocks and superblocks to minimize pipeline stalls and hide operation latency. |

---

## 5. AI Pass Selector

Instead of applying a static pass ordering regardless of program structure, CVM features an intelligent pass selector:

### How It Works:
1. **Feature Extraction (`ProgramFeatureExtractor`)**: Extracts 18 static features spanning:
   - *Structural*: loop count, loop nesting depth, basic block size, branch density, call density.
   - *Expressions*: constant operand fraction, copy instruction fraction, arithmetic intensity, memory ratio.
   - *Variables*: maximum simultaneous live variables, variable reuse rate, dead variable fraction.
2. **Heuristic Scoring Model (`PassScoringModel`)**: Scores each pass in $[0.0, 1.0]$ based on expected benefit formulas. For example:
   - LICM score scales with loops containing loop-invariant expressions.
   - Register allocation scores high when live variables fit well in virtual registers without excessive spills.
   - Vectorizer requires loops with linear array access and absence of loop-carried dependencies.
3. **Adaptive Sequence Generation (`PassSequenceGenerator`)**: Enforces prerequisite constraints (e.g., DCE requires Liveness, RegAlloc precedes Scheduler) while filtering out low-scoring passes ($< 0.2$).
4. **History Feedback Loop (`pass_history.json`)**: Persists optimization runs, recording static speedup and dynamic execution gains to refine future pass selections.

---

## 6. VM Verifier

Before any bytecode is executed, the static bytecode verifier (`verifier.py`) performs pre-execution validation:
- **Magic Number**: Asserts `"CVM1"` header bytes and declared instruction count.
- **Structural Integrity**: Validates valid opcodes, memory slot bounds, and non-NaN operands.
- **Control Flow Safety**: Ensures every branch target (`JMP`, `JZ`, `JNZ`, `CALL`) lands on a valid instruction boundary and verifies terminating `HALT` or `RET`.
- **Type Safety Simulation**: Abstractly executes the program with a type stack (`INT`, `FUNCREF`, `TOP`) ensuring no illegal operations (e.g. arithmetic on function references).
- **Merge Point Stack Depth**: Verifies that every control flow merge point receives identical operand stack depths from all predecessor paths.

---

## 7. Benchmarking Engine

The benchmark engine (`benchmark.py`) runs dual compilations:
1. **Baseline**: Direct compilation without optimizations or register allocation.
2. **Optimized**: Full AI-selected optimization pipeline with register allocation and scheduling.

### Verification Guarantee
The engine records VM printed outputs from both runs. If outputs diverge, a **`CORRECTNESS FAILURE`** is raised and no speedup is reported.

### Metrics Collected:
- **Static**: Instruction count change, basic block count, loop count, dead code removed, constants folded, copies eliminated, CSE replacements, LICM hoists, loops vectorized, register spills.
- **Dynamic**: VM instruction steps executed, median wall-clock time over 5 runs, peak operand stack depth, memory slot accesses (cache proxy), function calls.
- **Reports**: Generates formatted terminal box reports, structured JSON (`reports/<prog>_<ts>.json`), and academic Markdown tables (`reports/<prog>_<ts>.md`).

---

## 8. Usage

```bash
# Run full optimized pipeline (default: AI Pass Selector active)
python3 main.py examples/sum.ir

# Print basic blocks and control flow graph after optimization
python3 main.py examples/sum.ir --cfg

# Disassemble bytecode to readable assembly with virtual registers
python3 main.py examples/copy_chain.ir --dis

# Step-by-step VM execution trace
python3 main.py examples/fact.ir --trace

# Inspect AI Pass Selector feature extraction and scoring reasoning
python3 main.py examples/matmul.ir --explain

# Manual pass specification
python3 main.py examples/matmul.ir --passes DCE,LICM,RegAlloc

# Run dual-compilation benchmark and generate reports
python3 main.py examples/matmul.ir --benchmark

# Run complete regression test suite across all example programs
python3 benchmark.py --regression examples/

# Run end-to-end demo showcase
python3 demo.py
```

---

## 9. Examples

| Program | Source Path | Description | Key Passes Triggered |
|---|---|---|---|
| `constant_expr.ir` | [`examples/constant_expr.ir`](examples/constant_expr.ir) | Pure arithmetic expressions | `ConstantFolding`, `RegAlloc` |
| `copy_chain.ir` | [`examples/copy_chain.ir`](examples/copy_chain.ir) | Long copy assignment chain | `CopyPropagation`, `DCE`, `RegAlloc` |
| `matmul.ir` | [`examples/matmul.ir`](examples/matmul.ir) | 3×3 matrix multiplication | `LICM`, `StrengthRed`, `Unroll`, `RegAlloc` |
| `bubble_sort.ir` | [`examples/bubble_sort.ir`](examples/bubble_sort.ir) | Array bubble sorting | `LICM`, `ConstFold`, `Vectorizer` |
| `vector_add.ir` | [`examples/vector_add.ir`](examples/vector_add.ir) | Array vector addition | `Vectorizer` (4-wide SIMD), `RegAlloc` |
| `fibonacci.ir` | [`examples/fibonacci.ir`](examples/fibonacci.ir) | Recursive Fibonacci calculation | `RegAlloc`, `InstructionScheduler` |
| `sum.ir` | [`examples/sum.ir`](examples/sum.ir) | Loop sum $1 + 2 + \dots + 10$ | `ConstFold`, `RegAlloc`, `Scheduler` |
| `fact.ir` | [`examples/fact.ir`](examples/fact.ir) | Recursive factorial function | `ConstFold`, `RegAlloc`, `Scheduler` |
| `primes.ir` | [`examples/primes.ir`](examples/primes.ir) | Sieve / prime test loop | `StrengthRed`, `Unroll`, `RegAlloc` |

---

## 10. Module Coverage Table

| Module | Topic | Implemented | Implementation Location |
|---|---|:---:|---|
| **M5** | Code Optimization (DCE, ConstFold, CopyProp, CSE, LICM, Strength Reduction, Loop Unrolling) | ✅ | [`optimizer.py`](optimizer.py), [`dataflow.py`](dataflow.py), [`loops.py`](loops.py) |
| **M5** | Bytecode Verifier (safety, type consistency, stack depth) | ✅ | [`verifier.py`](verifier.py) |
| **M5** | AI-Based Pass Selection | ✅ | [`ai_selector.py`](ai_selector.py) |
| **M6** | Register Allocation (Chaitin-Briggs graph coloring) | ✅ | [`regalloc.py`](regalloc.py) |
| **M6** | Instruction Scheduling (list scheduling, DDG) | ✅ | [`scheduler.py`](scheduler.py) |
| **M7** | Vectorization (loop dependence analysis, vector IR) | ✅ | [`vectorizer.py`](vectorizer.py) |
| **M7** | Superblock Scheduling | ✅ | [`scheduler.py`](scheduler.py) |

---

## 11. Testing & Verification

All 126 unit tests across 10 test suites pass cleanly:
```bash
python3 tests.py
```
Output:
```
Ran 126 tests in 0.519s -> OK
```
Regression suite:
```bash
python3 benchmark.py --regression examples/
```
Output:
```
Summary: 10/10 benchmarks passed.
All benchmarks passed correctness and performance regression tests successfully!
```
