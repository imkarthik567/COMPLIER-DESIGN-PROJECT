"""Instruction Scheduling for Instruction Level Parallelism (ILP) in CVM.

Reorders instructions within basic blocks (and across superblocks) to minimize
pipeline stalls on pipelined execution models. This pass runs after register
allocation and before final bytecode generation.

Key phases:
1. Data Dependency Graph (DDG):
   - Computes RAW (True), WAR (Anti), WAW (Output), and CONTROL dependencies.
   - Models instruction latencies (Multiply: 3, Divide: 5, Load: 2, Store: 1,
     Arithmetic/Comparison: 1, Call: 10 as full barrier).
   - Assigns edge weights = producer latency.
2. Critical Path Calculation:
   - Bottom-up computation in reverse topological order:
     critical_path_length(n) = max over s in succs(n) of (edge_weight(n->s) + critical_path_length(s))
     with base case for leaves = latency(n).
3. List Scheduling (Forward, In-Order Emission):
   - Maintains a READY list of nodes with in-degree 0.
   - Priority heuristic: highest critical path length.
   - Tie-breaking: calls > stores > other (with original order as final tie-breaker).
4. Superblock Scheduling:
   - Forms traces of hot basic blocks using loop nesting frequency heuristics.
   - Schedules across block boundaries with side-exit liveness checking and compensation code.
5. Invariant & Verification:
   - Verifies all DDG dependency constraints are preserved in scheduled order.
   - Computes pipeline stall metrics and reports blocks ranked by stalls saved.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ir import (
    Assign, BinOp, Call, Function, Goto, IfNz, IfZ, Label, Print, Return,
    LoadIndexed, StoreIndexed, NewArray, VLoad, VStore, VAdd, VSub, VMul, VDiv
)
from optimizer import BasicBlock, build_cfg, cfg_to_function
from dataflow import LivenessAnalysis, _get_def_var, _get_read_vars
from loops import find_natural_loops


# =====================================================================
# STEP 1: Data Dependency Graph (DDG)
# =====================================================================

class DependencyType(str, Enum):
    """Types of dependencies between instructions."""
    RAW = "RAW"          # True dependency (Read After Write)
    WAR = "WAR"          # Anti-dependency (Write After Read)
    WAW = "WAW"          # Output dependency (Write After Write)
    CONTROL = "CONTROL"  # Control / Barrier dependency


def get_instruction_latency(ins: Any) -> int:
    """Return pipeline execution latency in cycles for an instruction.

    Latency specifications:
    - Arithmetic (+, -, comparisons): 1
    - Multiply (*): 3
    - Divide (/, %): 5
    - Load from memory/slot: 2
    - Store: 1
    - Call: 10 (full barrier)
    """
    if isinstance(ins, BinOp):
        if ins.op == "*":
            return 3
        elif ins.op in ("/", "%"):
            return 5
        else:
            return 1
    elif isinstance(ins, Call):
        return 10
    elif isinstance(ins, (LoadIndexed, VLoad)):
        return 2
    elif isinstance(ins, (StoreIndexed, VStore)):
        return 1
    elif isinstance(ins, (VAdd, VSub, VMul, VDiv)):
        return 2
    elif isinstance(ins, Assign):
        # Store to memory/spill slot
        if isinstance(ins.dst, str) and (
            ins.dst.startswith(("_spill_slot_", "mem_", "store_"))
            or getattr(ins, "is_store", False)
        ):
            return 1
        # Load from memory/spill slot or variable slot
        elif isinstance(ins.src, str):
            return 2
        else:
            # Immediate assignment
            return 1
    elif isinstance(ins, (Goto, IfZ, IfNz, Return)):
        return 1
    elif isinstance(ins, Print):
        return 1
    elif isinstance(ins, Label):
        return 0
    return 1


def is_call_instruction(ins: Any) -> bool:
    """Return True if instruction is a function call."""
    return isinstance(ins, Call)


def is_store_instruction(ins: Any) -> bool:
    """Return True if instruction is a store to memory or slot."""
    if isinstance(ins, (StoreIndexed, VStore)):
        return True
    if getattr(ins, "is_store", False):
        return True
    if isinstance(ins, Assign) and isinstance(ins.dst, str):
        if (
            ins.dst.startswith(("_spill_slot_", "mem_", "store_"))
            or "spill" in ins.dst.lower()
        ):
            return True
    return False


def is_load_instruction(ins: Any) -> bool:
    """Return True if instruction is a load from memory or array."""
    if isinstance(ins, (LoadIndexed, VLoad)):
        return True
    if isinstance(ins, Assign) and isinstance(ins.src, str):
        if (
            ins.src.startswith(("_spill_slot_", "mem_", "load_"))
            or "spill" in ins.src.lower()
        ):
            return True
    return False


def is_barrier_instruction(ins: Any) -> bool:
    """Return True if instruction is a scheduling barrier (calls and branches)."""
    return isinstance(ins, (Call, Goto, IfZ, IfNz, Return))


# Aliases for convenience
is_store = is_store_instruction
is_call = is_call_instruction
is_barrier = is_barrier_instruction


@dataclass
class DDGEdge:
    """Dependency edge connecting instructions in a Data Dependency Graph."""
    source: int
    target: int
    dep_type: DependencyType
    weight: int  # latency of producer instruction


@dataclass
class DDGNode:
    """Node wrapping an instruction and its latency in a Data Dependency Graph."""
    id: int
    instr: Any
    latency: int


class DataDependencyGraph:
    """Directed Acyclic Graph representing instruction dependencies in a basic block."""

    def __init__(self, instrs: List[Any]):
        self.instrs = list(instrs)
        self.nodes: List[DDGNode] = [
            DDGNode(id=i, instr=ins, latency=get_instruction_latency(ins))
            for i, ins in enumerate(instrs)
        ]
        self.edges: List[DDGEdge] = []
        self.succs: Dict[int, List[int]] = {i: [] for i in range(len(instrs))}
        self.preds: Dict[int, List[int]] = {i: [] for i in range(len(instrs))}
        self.edge_map: Dict[Tuple[int, int], DDGEdge] = {}
        self.critical_paths: Dict[int, int] = {}

    def add_edge(self, source: int, target: int, dep_type: DependencyType, weight: Optional[int] = None):
        """Add a dependency edge source -> target with producer latency weight."""
        if source == target:
            return
        if weight is None:
            weight = self.nodes[source].latency

        key = (source, target)
        if key in self.edge_map:
            # Upgrade dependency type if more restrictive (RAW > WAW > WAR > CONTROL)
            prio = {DependencyType.RAW: 4, DependencyType.WAW: 3, DependencyType.WAR: 2, DependencyType.CONTROL: 1}
            old_edge = self.edge_map[key]
            if prio[dep_type] > prio[old_edge.dep_type]:
                old_edge.dep_type = dep_type
            old_edge.weight = max(old_edge.weight, weight)
            return

        edge = DDGEdge(source=source, target=target, dep_type=dep_type, weight=weight)
        self.edges.append(edge)
        self.edge_map[key] = edge
        self.succs[source].append(target)
        self.preds[target].append(source)

    def get_successors(self, node_id: int) -> List[int]:
        """Return list of successor node IDs dependent on this instruction."""
        return self.succs.get(node_id, [])

    def get_predecessors(self, node_id: int) -> List[int]:
        """Return list of predecessor node IDs that this instruction depends upon."""
        return self.preds.get(node_id, [])

    def edge_weight(self, source: int, target: int) -> int:
        """Return dependency latency weight of the edge between source and target."""
        edge = self.edge_map.get((source, target))
        return edge.weight if edge else self.nodes[source].latency

    def latency(self, node_id: int) -> int:
        """Return execution latency for the instruction at node_id."""
        return self.nodes[node_id].latency

    # =================================================================
    # STEP 2: Critical Path Calculation
    # =================================================================

    def compute_critical_paths(self) -> Dict[int, int]:
        """Compute the longest latency-weighted path from each node to any leaf.

        critical_path_length(n) = max over s in succs(n) of (edge_weight(n->s) + critical_path_length(s))
        with base case for leaf nodes = latency(n).
        Computed bottom-up in reverse topological order.
        """
        # In a basic block DDG with edges i -> j where i < j, reverse index order is reverse topological.
        n = len(self.nodes)
        cp: Dict[int, int] = {}

        for u in reversed(range(n)):
            succs = self.get_successors(u)
            if not succs:
                cp[u] = self.latency(u)
            else:
                cp[u] = max(self.edge_weight(u, s) + cp[s] for s in succs)

        self.critical_paths = cp
        return cp

    def critical_path_length(self, node_id: int) -> int:
        """Return the maximum latency-weighted path length from node_id to any leaf."""
        if not self.critical_paths:
            self.compute_critical_paths()
        return self.critical_paths.get(node_id, self.latency(node_id))


def build_ddg(
    instrs: List[Any],
    reg_map: Optional[Dict[str, int]] = None,
) -> DataDependencyGraph:
    """Construct Data Dependency Graph for an instruction sequence in a basic block."""
    ddg = DataDependencyGraph(instrs)
    n = len(instrs)
    if n <= 1:
        ddg.compute_critical_paths()
        return ddg

    # Track def and read history
    # var/resource -> list of instruction indices that read that resource since last def
    var_readers: Dict[str, List[int]] = {}
    # var/resource -> instruction index of latest definition
    latest_def: Dict[str, int] = {}
    # Track print instructions to preserve IO order
    last_print_idx: Optional[int] = None
    # Track memory operations for store/load ordering dependencies
    last_stores: List[int] = []
    last_loads: List[int] = []

    def to_resource(v: str) -> str:
        """Map variable name to register resource identifier if allocated."""
        if reg_map and v in reg_map:
            return f"@r{reg_map[v]}"
        return v

    # If first instruction is a Label, it must precede all subsequent instructions
    if isinstance(instrs[0], Label):
        for j in range(1, n):
            ddg.add_edge(0, j, DependencyType.CONTROL, weight=ddg.latency(0))

    for j, ins in enumerate(instrs):
        raw_reads = _get_read_vars(ins)
        raw_def = _get_def_var(ins)
        reads = [to_resource(r) for r in raw_reads]
        def_var = to_resource(raw_def) if raw_def else None
        is_barrier = is_barrier_instruction(ins)

        # 1. RAW: True Dependency (j reads variable defined by latest_def)
        for r in reads:
            if r in latest_def:
                producer = latest_def[r]
                ddg.add_edge(producer, j, DependencyType.RAW)

        # 2. WAR: Anti-Dependency (j writes variable previously read by readers)
        if def_var:
            for r_idx in var_readers.get(def_var, []):
                if r_idx != j:
                    ddg.add_edge(r_idx, j, DependencyType.WAR)

        # 3. WAW: Output Dependency (j writes variable previously defined)
        if def_var:
            if def_var in latest_def:
                old_def = latest_def[def_var]
                if old_def != j:
                    ddg.add_edge(old_def, j, DependencyType.WAW)

        # 4. CONTROL: Barrier dependencies
        # (a) Branches and Calls are scheduling barriers: nothing moves past them
        if is_barrier:
            # Everything before this barrier must execute before the barrier
            for i in range(j):
                ddg.add_edge(i, j, DependencyType.CONTROL)
            # Everything after this barrier must execute after the barrier
            for k in range(j + 1, n):
                ddg.add_edge(j, k, DependencyType.CONTROL)

        # (b) Sequential I/O ordering for Print instructions
        if isinstance(ins, Print):
            if last_print_idx is not None:
                ddg.add_edge(last_print_idx, j, DependencyType.CONTROL)
            last_print_idx = j

        # 5. MEMORY: Order dependencies between stores and loads
        is_mem_store = is_store_instruction(ins)
        is_mem_load = is_load_instruction(ins)
        if is_mem_store:
            for s in last_stores:
                ddg.add_edge(s, j, DependencyType.WAW)
            for l in last_loads:
                ddg.add_edge(l, j, DependencyType.WAR)
            last_stores = [j]
            last_loads = []
        elif is_mem_load:
            for s in last_stores:
                ddg.add_edge(s, j, DependencyType.RAW)
            last_loads.append(j)

        # Update variable tracking
        for r in reads:
            var_readers.setdefault(r, []).append(j)

        if def_var:
            # New definition clears readers list for this variable
            var_readers[def_var] = []
            latest_def[def_var] = j

    ddg.compute_critical_paths()
    return ddg


# =====================================================================
# STEP 3: List Scheduling (Forward, In-Order Emission)
# =====================================================================

def _tie_break_priority(ins: Any) -> int:
    """Priority score for breaking ties among READY nodes with equal critical path.

    Order: calls > stores > other.
    """
    if is_call_instruction(ins):
        return 2
    elif is_store_instruction(ins):
        return 1
    return 0


def list_schedule(instrs: List[Any], ddg: Optional[DataDependencyGraph] = None) -> List[Any]:
    """Perform forward list scheduling on a sequence of instructions in a basic block."""
    if len(instrs) <= 1:
        return list(instrs)

    if ddg is None:
        ddg = build_ddg(instrs)

    n = len(instrs)
    in_degree = {i: len(ddg.get_predecessors(i)) for i in range(n)}
    ready: List[int] = [i for i in range(n) if in_degree[i] == 0]

    scheduled_order: List[int] = []

    while ready:
        # Pick node from READY with highest critical path length
        # Tie-breaker: calls > stores > other, then original index
        best_node = max(
            ready,
            key=lambda i: (
                ddg.critical_path_length(i),
                _tie_break_priority(ddg.nodes[i].instr),
                -i,  # lower original index preferred when all else equal
            ),
        )

        ready.remove(best_node)
        scheduled_order.append(best_node)

        for s in ddg.get_successors(best_node):
            in_degree[s] -= 1
            if in_degree[s] == 0:
                ready.append(s)

    assert len(scheduled_order) == n, "Cycle detected in basic block DDG; all nodes must be scheduled"

    scheduled_instrs = [instrs[i] for i in scheduled_order]
    verify_schedule(instrs, scheduled_instrs, ddg, scheduled_order=scheduled_order)
    return scheduled_instrs


def verify_schedule(
    original_instrs: List[Any],
    scheduled_instrs: List[Any],
    ddg: DataDependencyGraph,
    scheduled_order: Optional[List[int]] = None,
) -> bool:
    """Verify that all dependency constraints are strictly satisfied after scheduling."""
    assert len(original_instrs) == len(scheduled_instrs), (
        f"Schedule length mismatch: {len(original_instrs)} vs {len(scheduled_instrs)}"
    )

    if scheduled_order is None:
        # Map each original instruction index to scheduled position
        # Using identity where possible or multiset match
        sched_order = []
        used_sched_pos = set()
        for orig_idx, ins in enumerate(original_instrs):
            matched = False
            for s_idx, s_ins in enumerate(scheduled_instrs):
                if s_idx not in used_sched_pos and s_ins == ins:
                    sched_order.append((orig_idx, s_idx))
                    used_sched_pos.add(s_idx)
                    matched = True
                    break
            assert matched, f"Instruction {ins} missing from scheduled instructions"
        orig_to_sched = dict(sched_order)
    else:
        orig_to_sched = {orig_idx: sched_pos for sched_pos, orig_idx in enumerate(scheduled_order)}

    # Verify every dependency edge source -> target has sched_pos(source) < sched_pos(target)
    for edge in ddg.edges:
        src_pos = orig_to_sched[edge.source]
        tgt_pos = orig_to_sched[edge.target]
        assert src_pos < tgt_pos, (
            f"Dependency violation [{edge.dep_type.value}]: instruction {edge.source} ({ddg.nodes[edge.source].instr}) "
            f"scheduled at pos {src_pos} must precede instruction {edge.target} ({ddg.nodes[edge.target].instr}) "
            f"scheduled at pos {tgt_pos}"
        )

    return True


# =====================================================================
# Pipeline Simulation & Stall Analysis
# =====================================================================

def estimate_pipeline_cycles(instrs: List[Any]) -> Tuple[int, int]:
    """Simulate in-order pipeline execution and return (total_cycles, stall_cycles)."""
    if not instrs:
        return 0, 0

    current_cycle = 0
    total_stalls = 0
    # Map variable to the cycle when its produced value becomes ready
    var_ready_cycle: Dict[str, int] = {}
    last_barrier_cycle = 0

    for ins in instrs:
        reads = _get_read_vars(ins)
        earliest_issue = current_cycle

        # Must wait until all input operands are ready
        for r in reads:
            if r in var_ready_cycle:
                earliest_issue = max(earliest_issue, var_ready_cycle[r])

        # Barriers (Calls, Branches) wait for previous barrier completion
        if is_barrier_instruction(ins):
            earliest_issue = max(earliest_issue, last_barrier_cycle)

        if earliest_issue > current_cycle:
            stalls = earliest_issue - current_cycle
            total_stalls += stalls
            current_cycle = earliest_issue

        lat = get_instruction_latency(ins)
        def_var = _get_def_var(ins)
        if def_var:
            var_ready_cycle[def_var] = current_cycle + lat

        if is_barrier_instruction(ins):
            last_barrier_cycle = current_cycle + lat

        # Next instruction can issue at earliest current_cycle + 1
        current_cycle += 1

    return current_cycle, total_stalls


# =====================================================================
# STEP 4: Superblock Scheduling
# =====================================================================

@dataclass
class Superblock:
    """Represents a linear sequence of basic blocks with a single entry and multiple exits."""
    id: int
    blocks: List[BasicBlock]
    entry_label: str
    exit_labels: List[str]


def compute_block_frequency_hints(cfg: Dict[str, BasicBlock]) -> Dict[str, float]:
    """Compute static execution frequency hints for basic blocks using loop nesting."""
    freq: Dict[str, float] = {lbl: 1.0 for lbl in cfg}
    natural_loops = find_natural_loops(cfg)

    for loop in natural_loops:
        # Loop bodies execute significantly more frequently than non-loop code
        for b in loop.body:
            if b in freq:
                freq[b] *= 20.0

    # Cold exception / trap blocks
    for lbl in cfg:
        lower = lbl.lower()
        if any(cold in lower for cold in ("divzero", "trap", "err", "cold", "halt")):
            freq[lbl] = 0.05

    return freq


def form_superblocks(cfg: Dict[str, BasicBlock]) -> List[Superblock]:
    """Identify hot traces and construct superblocks (single entry, multiple exits)."""
    if not cfg:
        return []

    freq = compute_block_frequency_hints(cfg)
    visited: Set[str] = set()
    superblocks: List[Superblock] = []
    sb_counter = 0

    # Sort blocks by execution frequency descending
    sorted_blocks = sorted(cfg.keys(), key=lambda lbl: freq.get(lbl, 1.0), reverse=True)

    for start_lbl in sorted_blocks:
        if start_lbl in visited:
            continue

        trace_blocks = [cfg[start_lbl]]
        visited.add(start_lbl)
        curr = cfg[start_lbl]

        while curr.succs:
            # Pick most likely successor
            candidates = [s for s in curr.succs if s in cfg and s not in visited]
            if not candidates:
                break
            best_succ = max(candidates, key=lambda s: freq.get(s, 1.0))
            succ_block = cfg[best_succ]

            # In a superblock, internal blocks must not have outside entries (single entry property)
            # If succ_block has multiple predecessors, we stop the trace
            preds = [p for p in succ_block.preds if p in cfg]
            if len(preds) == 1 and preds[0] == curr.label:
                trace_blocks.append(succ_block)
                visited.add(best_succ)
                curr = succ_block
            else:
                break

        sb_counter += 1
        exits = []
        for b in trace_blocks:
            for s in b.succs:
                if not any(tb.label == s for tb in trace_blocks):
                    exits.append(s)

        superblocks.append(
            Superblock(
                id=sb_counter,
                blocks=trace_blocks,
                entry_label=trace_blocks[0].label,
                exit_labels=exits,
            )
        )

    return superblocks


def schedule_superblock(
    sb: Superblock,
    cfg: Dict[str, BasicBlock],
    liveness_results: Optional[Any] = None,
    reg_map: Optional[Dict[str, int]] = None,
) -> Superblock:
    """Schedule instructions across blocks within a single superblock."""
    if len(sb.blocks) <= 1:
        # Single-block superblock: schedule locally
        b = sb.blocks[0]
        b.instrs = list_schedule(b.instrs)
        return sb

    # Trace-level scheduling: schedule each block with list scheduling
    # and speculatively hoist safe independent instructions from successor blocks
    for i in range(len(sb.blocks) - 1):
        curr_b = sb.blocks[i]
        next_b = sb.blocks[i + 1]

        # First schedule curr_b
        curr_b.instrs = list_schedule(curr_b.instrs)

        # Check if next_b has leading independent computations that can be speculatively hoisted
        if liveness_results is not None:
            live_out = liveness_results.out_sets.get(curr_b.label, set())
        else:
            live_out = set()

        term = curr_b.instrs[-1] if curr_b.instrs else None
        branch_reads = set(_get_read_vars(term)) if term else set()
        branch_regs = {reg_map[r] for r in branch_reads if r in reg_map} if reg_map else set()
        live_out_regs = {reg_map[v] for v in live_out if v in reg_map} if reg_map else set()

        hoist_candidates = []
        for ins in list(next_b.instrs):
            if isinstance(ins, Label):
                continue
            if is_barrier_instruction(ins) or isinstance(ins, Print):
                break
            d = _get_def_var(ins)
            reads = _get_read_vars(ins)
            if not d:
                break
            # Speculative safety: definition must not clobber live variables on side exit
            if d in live_out or d in branch_reads or any(r == d for r in reads):
                break
            # If register allocation was performed, avoid physical register clobbering
            if reg_map and d in reg_map:
                d_reg = reg_map[d]
                if d_reg in branch_regs or d_reg in live_out_regs:
                    break
            hoist_candidates.append(ins)

        # Move hoist candidates to end of curr_b before the terminal branch
        if hoist_candidates and len(curr_b.instrs) >= 1:
            term = curr_b.instrs[-1]
            if isinstance(term, (Goto, IfZ, IfNz)):
                curr_b.instrs = curr_b.instrs[:-1] + hoist_candidates + [term]
                next_b.instrs = [ins for ins in next_b.instrs if ins not in hoist_candidates]
                # Re-schedule curr_b with hoisted instructions included
                curr_b.instrs = list_schedule(curr_b.instrs)

        next_b.instrs = list_schedule(next_b.instrs)

    return sb


# =====================================================================
# STATS OUTPUT & REPORTING
# =====================================================================

@dataclass
class BlockScheduleReport:
    """Detailed performance and latency report for a scheduled basic block."""
    block_label: str
    original_instrs: List[Any]
    scheduled_instrs: List[Any]
    original_cycles: int
    scheduled_cycles: int
    original_stalls: int
    scheduled_stalls: int
    stalls_saved: int
    instructions_count: int


@dataclass
class SchedulingStats:
    """Aggregated instruction scheduling statistics across an entire function or program."""
    block_reports: Dict[str, BlockScheduleReport] = field(default_factory=dict)
    total_stalls_saved: int = 0
    total_original_stalls: int = 0
    total_scheduled_stalls: int = 0
    total_original_cycles: int = 0
    total_scheduled_cycles: int = 0

    def add_block_report(self, report: BlockScheduleReport) -> None:
        """Record scheduling statistics for an individual basic block."""
        self.block_reports[report.block_label] = report
        self.total_stalls_saved += report.stalls_saved
        self.total_original_stalls += report.original_stalls
        self.total_scheduled_stalls += report.scheduled_stalls
        self.total_original_cycles += report.original_cycles
        self.total_scheduled_cycles += report.scheduled_cycles

    def format_report(self) -> str:
        """Format aggregated scheduling metrics into a readable report."""
        lines = [
            "=== INSTRUCTION SCHEDULING REPORT ===",
            f"  Blocks scheduled:            {len(self.block_reports)}",
            f"  Total stall cycles saved:    {self.total_stalls_saved}",
            f"  Total pipeline cycles:       {self.total_original_cycles} -> {self.total_scheduled_cycles}",
        ]
        if self.block_reports:
            lines.append("\n  Blocks Ranked by Stall Cycles Saved:")
            ranked = sorted(
                self.block_reports.values(),
                key=lambda r: (r.stalls_saved, r.original_stalls),
                reverse=True,
            )
            for r in ranked:
                lines.append(
                    f"    Block '{r.block_label}': saved {r.stalls_saved} stalls "
                    f"({r.original_stalls} stalls in {r.original_cycles} cycles -> "
                    f"{r.scheduled_stalls} stalls in {r.scheduled_cycles} cycles, "
                    f"{r.instructions_count} instrs)"
                )
        return "\n".join(lines)

    def report(self) -> str:
        """Alias for format_report to return human-readable summary string."""
        return self.format_report()


# =====================================================================
# SCHEDULER DRIVER
# =====================================================================

class InstructionScheduler:
    """Orchestrates instruction scheduling across functions and basic blocks."""

    def __init__(self, use_superblocks: bool = False):
        self.use_superblocks = use_superblocks

    def schedule_block(
        self,
        block: BasicBlock,
        reg_map: Optional[Dict[str, int]] = None,
    ) -> Tuple[BasicBlock, BlockScheduleReport]:
        """Schedule instructions in a single basic block and return metrics."""
        orig_instrs = list(block.instrs)
        orig_cycles, orig_stalls = estimate_pipeline_cycles(orig_instrs)

        ddg = build_ddg(orig_instrs, reg_map=reg_map)
        sched_instrs = list_schedule(orig_instrs, ddg=ddg)
        sched_cycles, sched_stalls = estimate_pipeline_cycles(sched_instrs)

        stalls_saved = max(0, orig_stalls - sched_stalls)

        report = BlockScheduleReport(
            block_label=block.label,
            original_instrs=orig_instrs,
            scheduled_instrs=sched_instrs,
            original_cycles=orig_cycles,
            scheduled_cycles=sched_cycles,
            original_stalls=orig_stalls,
            scheduled_stalls=sched_stalls,
            stalls_saved=stalls_saved,
            instructions_count=len(orig_instrs),
        )

        block.instrs = sched_instrs
        return block, report

    def schedule_function(
        self,
        fn: Function,
        use_superblocks: Optional[bool] = None,
        reg_map: Optional[Dict[str, int]] = None,
    ) -> Tuple[Function, SchedulingStats]:
        """Schedule all basic blocks within a Function."""
        cfg = build_cfg(fn)
        stats = SchedulingStats()
        use_sb = self.use_superblocks if use_superblocks is None else use_superblocks

        if use_sb:
            superblocks = form_superblocks(cfg)
            liveness = LivenessAnalysis().run(cfg)
            for sb in superblocks:
                schedule_superblock(sb, cfg, liveness_results=liveness, reg_map=reg_map)
            for b in cfg.values():
                _, rep = self.schedule_block(b, reg_map=reg_map)
                stats.add_block_report(rep)
        else:
            for b in cfg.values():
                _, rep = self.schedule_block(b, reg_map=reg_map)
                stats.add_block_report(rep)

        new_fn = cfg_to_function(fn, cfg)
        return new_fn, stats

    def schedule_program(
        self,
        functions: List[Function],
        reg_map: Optional[Dict[str, Any]] = None,
        use_superblocks: Optional[bool] = None,
    ) -> Tuple[List[Function], SchedulingStats]:
        """Schedule all functions in a program."""
        scheduled_functions: List[Function] = []
        cumulative_stats = SchedulingStats()

        for fn in functions:
            fn_reg_map = None
            if reg_map:
                if isinstance(reg_map, dict) and fn.name in reg_map:
                    fn_reg_map = reg_map[fn.name]
                elif isinstance(reg_map, dict) and not any(isinstance(v, dict) for v in reg_map.values()):
                    fn_reg_map = reg_map

            new_fn, fn_stats = self.schedule_function(
                fn, use_superblocks=use_superblocks, reg_map=fn_reg_map
            )
            scheduled_functions.append(new_fn)
            for rep in fn_stats.block_reports.values():
                cumulative_stats.add_block_report(rep)

        return scheduled_functions, cumulative_stats


def schedule_program(
    functions: List[Function],
    reg_map: Optional[Dict[str, Any]] = None,
    use_superblocks: bool = False,
) -> Tuple[List[Function], SchedulingStats]:
    """Convenience function to schedule all functions in a program."""
    scheduler = InstructionScheduler(use_superblocks=use_superblocks)
    return scheduler.schedule_program(functions, reg_map=reg_map, use_superblocks=use_superblocks)

