"""Chaitin-Briggs Graph-Coloring Register Allocator for CVM.

Provides:
- N_REGS = 8 virtual registers (R0-R7).
- Step 1: Interference Graph construction via backward liveness analysis.
- Step 2: Conservative Coalescing (George-Briggs criteria) to eliminate copies.
- Step 3: Graph Coloring (Chaitin-Briggs) with simplification and optimistic spilling.
- Step 4: Spill Code Insertion with SpillSlotAllocator and bounded re-allocation.
- Step 5: Register-aware code generation and statistics.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from optimizer import BasicBlock, build_cfg
from dataflow import DataFlowResults, LivenessAnalysis, _get_def_var, _get_read_vars
from ir import (
    Assign, BinOp, Call, Function, Goto, IfNz, IfZ, Label, Print, Return,
    LoadIndexed, StoreIndexed, NewArray, VLoad, VStore, VAdd, VSub, VMul, VDiv
)

N_REGS = 8


@dataclass
class RegAllocStats:
    """Statistics and metrics from graph-coloring register allocation."""
    total_variables: int = 0
    assigned_to_registers: int = 0
    spills: int = 0
    copies_coalesced: int = 0
    estimated_instruction_reduction: int = 0

    def format_report(self) -> str:
        """Format register allocation metrics into a readable report."""
        lines = [
            "=== REGISTER ALLOCATION REPORT ===",
            f"  Total variables:             {self.total_variables}",
            f"  Assigned to registers:       {self.assigned_to_registers}",
            f"  Spills to memory:            {self.spills}",
            f"  Copies coalesced:            {self.copies_coalesced}",
            f"  Estimated instrs saved:      {self.estimated_instruction_reduction}",
        ]
        return "\n".join(lines)

    def report(self) -> str:
        """Alias for format_report to return human-readable summary string."""
        return self.format_report()


# =====================================================================
# STEP 1: Interference Graph
# =====================================================================

class InterferenceGraph:
    """Interference graph represented as an adjacency set."""

    def __init__(self, variables: Optional[Iterable[str]] = None):
        self.adj: Dict[str, Set[str]] = {}
        if variables:
            for v in variables:
                self.adj.setdefault(v, set())

    def add_node(self, var: str) -> None:
        """Add a variable node to the interference graph."""
        self.adj.setdefault(var, set())

    def remove_node(self, var: str) -> None:
        """Remove a variable node and all its incident edges."""
        if var in self.adj:
            for nbr in list(self.adj[var]):
                self.adj[nbr].discard(var)
            del self.adj[var]

    def add_edge(self, u: str, v: str) -> None:
        """Add an undirected interference edge between variables u and v."""
        if u != v:
            self.adj.setdefault(u, set()).add(v)
            self.adj.setdefault(v, set()).add(u)

    def remove_edge(self, u: str, v: str) -> None:
        """Remove the interference edge between variables u and v."""
        if u in self.adj:
            self.adj[u].discard(v)
        if v in self.adj:
            self.adj[v].discard(u)

    def degree(self, var: str) -> int:
        """Return the degree (number of interfering neighbors) of variable var."""
        return len(self.adj.get(var, set()))

    def neighbors(self, var: str) -> Set[str]:
        """Return the set of interfering neighbors for variable var."""
        return set(self.adj.get(var, set()))

    def nodes(self) -> Set[str]:
        """Return the set of all variable nodes in the graph."""
        return set(self.adj.keys())

    def clone(self) -> "InterferenceGraph":
        """Return an independent deep copy of the interference graph."""
        g = InterferenceGraph()
        for u, nbrs in self.adj.items():
            g.adj[u] = set(nbrs)
        return g


def build_interference_graph(
    fn: Function,
    cfg: Dict[str, BasicBlock],
    liveness_results: Optional[DataFlowResults] = None,
) -> Tuple[InterferenceGraph, List[Tuple[str, str]], Dict[str, int]]:
    """Construct interference graph from CFG and liveness analysis.

    Two variables interfere if they are simultaneously live at any program point.
    For a copy instruction x = y, x and y do not interfere by virtue of this definition.
    """
    if liveness_results is None:
        liveness_results = LivenessAnalysis().run(cfg)

    all_vars: Set[str] = set()
    uses_and_defs: Dict[str, int] = {}
    copy_pairs: List[Tuple[str, str]] = []

    for ins in fn.body:
        d = _get_def_var(ins)
        if d and not d.startswith("_spill_slot_"):
            all_vars.add(d)
            uses_and_defs[d] = uses_and_defs.get(d, 0) + 1
        for r in _get_read_vars(ins):
            if not r.startswith("_spill_slot_"):
                all_vars.add(r)
                uses_and_defs[r] = uses_and_defs.get(r, 0) + 1
        if (
            isinstance(ins, Assign)
            and isinstance(ins.src, str)
            and ins.src != ins.dst
            and not ins.src.startswith("_spill_slot_")
            and not ins.dst.startswith("_spill_slot_")
        ):
            copy_pairs.append((ins.dst, ins.src))

    for p in fn.params:
        if not p.startswith("_spill_slot_"):
            all_vars.add(p)
            uses_and_defs.setdefault(p, 1)

    graph = InterferenceGraph(all_vars)

    for lbl, block in cfg.items():
        live: Set[str] = {v for v in liveness_results.out_sets.get(lbl, set()) if not v.startswith("_spill_slot_")}
        for ins in reversed(block.instrs):
            d = _get_def_var(ins)
            if d and not d.startswith("_spill_slot_"):
                copy_src = ins.src if isinstance(ins, Assign) and isinstance(ins.src, str) else None
                for v in live:
                    if v != d and v != copy_src:
                        graph.add_edge(d, v)
                live.discard(d)
            for r in _get_read_vars(ins):
                if not r.startswith("_spill_slot_"):
                    live.add(r)

    return graph, copy_pairs, uses_and_defs


# =====================================================================
# STEP 2: Coalescing (George-Briggs Style)
# =====================================================================

def coalesce_copies(
    graph: InterferenceGraph,
    copy_pairs: List[Tuple[str, str]],
    K: int = N_REGS,
    precolored: Optional[Dict[str, int]] = None,
) -> Tuple[Dict[str, str], int]:
    """Conservatively coalesce copy candidates x = y.

    Uses George-Briggs criterion: coalesce only if the merged node has fewer
    than K high-degree neighbors, avoiding uncolorable graphs.
    """
    alias: Dict[str, str] = {}

    def get_alias(x: str) -> str:
        """Find canonical representative variable for coalesced alias group."""
        curr = x
        while curr in alias:
            curr = alias[curr]
        return curr

    coalesced_count = 0
    changed = True
    while changed:
        changed = False
        new_copy_pairs = []
        for u_raw, v_raw in copy_pairs:
            u = get_alias(u_raw)
            v = get_alias(v_raw)
            if u == v:
                continue

            if precolored:
                u_pre = u in precolored
                v_pre = v in precolored
                if u_pre and v_pre:
                    # Cannot coalesce two distinct precolored registers
                    if precolored[u] != precolored[v]:
                        new_copy_pairs.append((u, v))
                        continue
                elif v_pre and not u_pre:
                    # Keep the precolored node as the representative
                    u, v = v, u

            if v in graph.neighbors(u):
                new_copy_pairs.append((u, v))
                continue

            # Briggs test: merged node has fewer than K high-degree neighbors
            combined_nbrs = (graph.neighbors(u) | graph.neighbors(v)) - {u, v}
            high_deg_count = sum(1 for t in combined_nbrs if graph.degree(t) >= K)

            # George test: for all t in neighbors(v), degree(t) < K or t in neighbors(u)
            george_ok = all(graph.degree(t) < K or t in graph.neighbors(u) for t in graph.neighbors(v))

            if high_deg_count < K or george_ok:
                for nbr in list(graph.neighbors(v)):
                    if nbr != u:
                        graph.add_edge(u, nbr)
                graph.remove_node(v)
                alias[v] = u
                coalesced_count += 1
                changed = True
            else:
                new_copy_pairs.append((u, v))

        copy_pairs = new_copy_pairs

    final_alias = {k: get_alias(k) for k in alias}
    return final_alias, coalesced_count


# =====================================================================
# STEP 3: Graph Coloring (Chaitin-Briggs)
# =====================================================================

def color_graph(
    graph: InterferenceGraph,
    uses_and_defs: Dict[str, int],
    K: int = N_REGS,
    precolored: Optional[Dict[str, int]] = None,
) -> Tuple[Dict[str, int], Set[str]]:
    """Color the interference graph using Chaitin-Briggs with optimistic select.

    Returns:
        (colors: { var -> register_id (0..K-1) }, actual_spills: Set[var])
    """
    active_graph = graph.clone()
    select_stack: List[str] = []
    precolored_set = set(precolored.keys()) if precolored else set()

    # Simplification & Potential Spill Phase
    while True:
        candidates = [n for n in active_graph.nodes() if n not in precolored_set]
        if not candidates:
            break

        low_deg = [n for n in candidates if active_graph.degree(n) < K]
        if low_deg:
            chosen = min(low_deg, key=lambda n: active_graph.degree(n))
            active_graph.remove_node(chosen)
            select_stack.append(chosen)
        else:
            # Potential spill: spill priority = (uses + defs) / degree
            chosen = min(
                candidates,
                key=lambda n: (uses_and_defs.get(n, 1)) / max(1, active_graph.degree(n)),
            )
            active_graph.remove_node(chosen)
            select_stack.append(chosen)

    # Select Phase (pop and color greedily)
    colors: Dict[str, int] = dict(precolored) if precolored else {}
    actual_spills: Set[str] = set()

    while select_stack:
        node = select_stack.pop()
        used_colors = {
            colors[nbr] for nbr in graph.neighbors(node) if nbr in colors
        }
        assigned_color = None
        for c in range(K):
            if c not in used_colors:
                assigned_color = c
                break

        if assigned_color is not None:
            colors[node] = assigned_color
        else:
            actual_spills.add(node)

    return colors, actual_spills


# =====================================================================
# STEP 4: Spill Code Insertion
# =====================================================================

class SpillSlotAllocator:
    """Allocates unique memory spill slots starting from a given index."""

    def __init__(self, start_slot: int = 0):
        self.next_slot = start_slot
        self.var_to_slot: Dict[str, int] = {}

    def alloc_slot(self, var: str) -> int:
        """Allocate or retrieve an existing stack frame spill slot for variable var."""
        if var not in self.var_to_slot:
            self.var_to_slot[var] = self.next_slot
            self.next_slot += 1
        return self.var_to_slot[var]

    def slot_for(self, var: str) -> Optional[int]:
        """Return allocated stack spill slot index for var, or None if unassigned."""
        return self.var_to_slot.get(var)


def _replace_uses(ins: Any, rep_map: Dict[str, str]) -> Any:
    """Replace variable uses in an instruction."""
    if isinstance(ins, Assign):
        new_src = rep_map.get(ins.src, ins.src) if isinstance(ins.src, str) else ins.src
        return Assign(dst=ins.dst, src=new_src)
    elif isinstance(ins, BinOp):
        new_a = rep_map.get(ins.a, ins.a) if isinstance(ins.a, str) else ins.a
        new_b = rep_map.get(ins.b, ins.b) if isinstance(ins.b, str) else ins.b
        return BinOp(dst=ins.dst, op=ins.op, a=new_a, b=new_b)
    elif isinstance(ins, IfZ):
        new_cond = rep_map.get(ins.cond, ins.cond) if isinstance(ins.cond, str) else ins.cond
        return IfZ(cond=new_cond, target=ins.target)
    elif isinstance(ins, IfNz):
        new_cond = rep_map.get(ins.cond, ins.cond) if isinstance(ins.cond, str) else ins.cond
        return IfNz(cond=new_cond, target=ins.target)
    elif isinstance(ins, Print):
        new_arg = rep_map.get(ins.arg, ins.arg) if isinstance(ins.arg, str) else ins.arg
        return Print(arg=new_arg)
    elif isinstance(ins, Return):
        if ins.arg is not None and isinstance(ins.arg, str):
            return Return(arg=rep_map.get(ins.arg, ins.arg))
        return ins
    elif isinstance(ins, Call):
        new_args = [rep_map.get(a, a) if isinstance(a, str) else a for a in ins.args]
        return Call(dst=ins.dst, func=ins.func, args=new_args)
    elif isinstance(ins, LoadIndexed):
        new_arr = rep_map.get(ins.array, ins.array) if isinstance(ins.array, str) else ins.array
        new_idx = rep_map.get(ins.index, ins.index) if isinstance(ins.index, str) else ins.index
        return LoadIndexed(dst=ins.dst, array=new_arr, index=new_idx)
    elif isinstance(ins, StoreIndexed):
        new_arr = rep_map.get(ins.array, ins.array) if isinstance(ins.array, str) else ins.array
        new_idx = rep_map.get(ins.index, ins.index) if isinstance(ins.index, str) else ins.index
        new_src = rep_map.get(ins.src, ins.src) if isinstance(ins.src, str) else ins.src
        return StoreIndexed(array=new_arr, index=new_idx, src=new_src)
    elif isinstance(ins, NewArray):
        new_size = rep_map.get(ins.size, ins.size) if isinstance(ins.size, str) else ins.size
        return NewArray(dst=ins.dst, size=new_size)
    elif isinstance(ins, VLoad):
        new_arr = rep_map.get(ins.array, ins.array) if isinstance(ins.array, str) else ins.array
        new_idx = rep_map.get(ins.base_index, ins.base_index) if isinstance(ins.base_index, str) else ins.base_index
        return VLoad(dst=ins.dst, array=new_arr, base_index=new_idx, width=ins.width)
    elif isinstance(ins, VStore):
        new_arr = rep_map.get(ins.array, ins.array) if isinstance(ins.array, str) else ins.array
        new_idx = rep_map.get(ins.base_index, ins.base_index) if isinstance(ins.base_index, str) else ins.base_index
        new_src = rep_map.get(ins.src, ins.src) if isinstance(ins.src, str) else ins.src
        return VStore(array=new_arr, base_index=new_idx, src=new_src, width=ins.width)
    elif isinstance(ins, (VAdd, VSub, VMul, VDiv)):
        cls = type(ins)
        new_a = rep_map.get(ins.a, ins.a) if isinstance(ins.a, str) else ins.a
        new_b = rep_map.get(ins.b, ins.b) if isinstance(ins.b, str) else ins.b
        return cls(dst=ins.dst, a=new_a, b=new_b)
    return ins


def _replace_def(ins: Any, new_dst: str) -> Any:
    """Replace definition destination in an instruction."""
    if isinstance(ins, Assign):
        return Assign(dst=new_dst, src=ins.src)
    elif isinstance(ins, BinOp):
        return BinOp(dst=new_dst, op=ins.op, a=ins.a, b=ins.b)
    elif isinstance(ins, Call):
        return Call(dst=new_dst, func=ins.func, args=ins.args)
    elif isinstance(ins, LoadIndexed):
        return LoadIndexed(dst=new_dst, array=ins.array, index=ins.index)
    elif isinstance(ins, NewArray):
        return NewArray(dst=new_dst, size=ins.size)
    elif isinstance(ins, VLoad):
        return VLoad(dst=new_dst, array=ins.array, base_index=ins.base_index, width=ins.width)
    elif isinstance(ins, (VAdd, VSub, VMul, VDiv)):
        cls = type(ins)
        return cls(dst=new_dst, a=ins.a, b=ins.b)
    return ins


def insert_spill_code(
    fn: Function,
    spilled_vars: Set[str],
    spill_alloc: SpillSlotAllocator,
    iteration: int = 1,
) -> Function:
    """Insert spill load before each use and spill store after each definition."""
    new_body = []
    tmp_counter = 0

    for ins in fn.body:
        reads = _get_read_vars(ins)
        spilled_reads = [r for r in reads if r in spilled_vars]

        reload_map: Dict[str, str] = {}
        for r in spilled_reads:
            slot_num = spill_alloc.alloc_slot(r)
            tmp_counter += 1
            reload_var = f"{r}_load_s{slot_num}_i{iteration}_{tmp_counter}"
            reload_map[r] = reload_var
            spill_slot_var = f"_spill_slot_{slot_num}"
            new_body.append(Assign(dst=reload_var, src=spill_slot_var))

        rewritten_ins = ins
        if reload_map:
            rewritten_ins = _replace_uses(ins, reload_map)

        d = _get_def_var(rewritten_ins)
        if d in spilled_vars:
            slot_num = spill_alloc.alloc_slot(d)
            tmp_counter += 1
            def_var = f"{d}_store_s{slot_num}_i{iteration}_{tmp_counter}"
            rewritten_ins = _replace_def(rewritten_ins, def_var)
            new_body.append(rewritten_ins)
            spill_slot_var = f"_spill_slot_{slot_num}"
            new_body.append(Assign(dst=spill_slot_var, src=def_var))
        else:
            new_body.append(rewritten_ins)

    return Function(name=fn.name, params=list(fn.params), body=new_body, line=fn.line)


# =====================================================================
# STEP 5: Register Allocator Driver
# =====================================================================

class RegisterAllocator:
    """Full Chaitin-Briggs graph coloring register allocator."""

    def __init__(self, K: int = N_REGS, max_spill_iterations: int = 3):
        self.K = K
        self.max_spill_iterations = max_spill_iterations

    def allocate(self, fn: Function) -> Tuple[Dict[str, int], Function, RegAllocStats]:
        """Perform register allocation on a single function."""
        current_fn = fn
        spill_alloc = SpillSlotAllocator()
        total_spills = 0
        total_coalesced = 0
        final_colors: Dict[str, int] = {}

        for iteration in range(1, self.max_spill_iterations + 1):
            cfg = build_cfg(current_fn)
            liveness = LivenessAnalysis().run(cfg)
            graph, copy_pairs, uses_and_defs = build_interference_graph(current_fn, cfg, liveness)

            # Precolor incoming function parameters to ABI argument registers (R0, R1, ...)
            precolored = {p: i for i, p in enumerate(current_fn.params) if i < self.K}

            # Coalescing
            alias_map, coalesced_count = coalesce_copies(graph, copy_pairs, K=self.K, precolored=precolored)
            total_coalesced += coalesced_count

            # Coloring
            colors, actual_spills = color_graph(graph, uses_and_defs, K=self.K, precolored=precolored)

            # Map aliases to allocated colors
            for v, rep in alias_map.items():
                if rep in colors:
                    colors[v] = colors[rep]

            final_colors = colors
            if not actual_spills:
                break

            total_spills += len(actual_spills)
            current_fn = insert_spill_code(current_fn, actual_spills, spill_alloc, iteration=iteration)

        # Collect distinct variables
        all_vars: Set[str] = set()
        for ins in current_fn.body:
            d = _get_def_var(ins)
            if d:
                all_vars.add(d)
            for r in _get_read_vars(ins):
                all_vars.add(r)
        for p in current_fn.params:
            all_vars.add(p)

        assigned_count = sum(1 for v in all_vars if v in final_colors)

        # Estimate instruction savings:
        # - Coalesced copies: eliminated completely (saves 2 stack instrs: LOAD + STORE)
        # - RLOAD Rd, Rs: saves 1 instr vs LOAD + STORE
        # - RLOAD_IMM Rd, v: saves 1 instr vs PUSH + STORE
        instr_savings = 2 * total_coalesced
        for ins in current_fn.body:
            if isinstance(ins, Assign):
                if isinstance(ins.src, int) and ins.dst in final_colors:
                    instr_savings += 1
                elif isinstance(ins.src, str) and ins.dst in final_colors and ins.src in final_colors:
                    if final_colors[ins.dst] != final_colors[ins.src]:
                        instr_savings += 1

        stats = RegAllocStats(
            total_variables=len(all_vars),
            assigned_to_registers=assigned_count,
            spills=total_spills,
            copies_coalesced=total_coalesced,
            estimated_instruction_reduction=instr_savings,
        )

        return final_colors, current_fn, stats


def allocate_registers_for_program(
    functions: List[Function], K: int = N_REGS
) -> Tuple[Dict[str, Dict[str, int]], List[Function], RegAllocStats]:
    """Perform register allocation across all functions in a program."""
    program_reg_map: Dict[str, Dict[str, int]] = {}
    rewritten_functions: List[Function] = []
    cumulative_stats = RegAllocStats()

    allocator = RegisterAllocator(K=K)
    for fn in functions:
        colors, new_fn, fn_stats = allocator.allocate(fn)
        program_reg_map[fn.name] = colors
        rewritten_functions.append(new_fn)

        cumulative_stats.total_variables += fn_stats.total_variables
        cumulative_stats.assigned_to_registers += fn_stats.assigned_to_registers
        cumulative_stats.spills += fn_stats.spills
        cumulative_stats.copies_coalesced += fn_stats.copies_coalesced
        cumulative_stats.estimated_instruction_reduction += fn_stats.estimated_instruction_reduction

    return program_reg_map, rewritten_functions, cumulative_stats
