"""Loop Analysis for CVM Compiler Backend.

Provides:
1. Cooper-Harvey-Kennedy Dominator Tree algorithm (idom, dom, dom_tree)
2. Natural Loop Detection (header, body, back_edge, exit_blocks)
3. Loop Nesting Tree (nested loops, inner vs outer loops, nesting depth)
4. Primary Induction Variable Detection
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ir import Assign, BinOp, Call


def _get_def_var(ins) -> Optional[str]:
    """Extract variable name defined/assigned by an instruction, if any."""
    if isinstance(ins, (Assign, BinOp)):
        return ins.dst
    if isinstance(ins, Call):
        return ins.dst
    return None



@dataclass
class NaturalLoop:
    """Represents a natural loop in the control flow graph."""
    header: str
    body: Set[str]
    back_edge: Tuple[str, str]  # (tail_block, header_block)
    exit_blocks: Set[str] = field(default_factory=set)

    def is_nested_in(self, other: "NaturalLoop") -> bool:
        """Returns True if this loop is strictly nested inside other."""
        return self.body < other.body and self != other


@dataclass
class LoopNestNode:
    """A node in the loop nesting tree."""
    loop: Optional[NaturalLoop]  # None for the synthetic root (entire function)
    children: List["LoopNestNode"] = field(default_factory=list)
    parent: Optional["LoopNestNode"] = None

    @property
    def is_root(self) -> bool:
        return self.loop is None

    @property
    def is_innermost(self) -> bool:
        return len(self.children) == 0 and self.loop is not None


class LoopNestingTree:
    """Tree representing loop containment hierarchy."""
    def __init__(self, loops: List[NaturalLoop]):
        self.root = LoopNestNode(loop=None)
        self.all_nodes: List[LoopNestNode] = []
        self._build_tree(loops)

    def _build_tree(self, loops: List[NaturalLoop]):
        # Sort descending by body size: outer loops before inner loops
        sorted_loops = sorted(loops, key=lambda l: len(l.body), reverse=True)
        nodes = [LoopNestNode(loop=l) for l in sorted_loops]
        self.all_nodes = nodes

        for i, node in enumerate(nodes):
            parent_node = None
            # Find the most immediate (smallest) enclosing loop
            for other in reversed(nodes[:i]):
                if node.loop.is_nested_in(other.loop):
                    parent_node = other
                    break
            if parent_node is None:
                parent_node = self.root
            node.parent = parent_node
            parent_node.children.append(node)

    def get_innermost_loops(self) -> List[NaturalLoop]:
        """Return all loops that contain no sub-loops."""
        return [node.loop for node in self.all_nodes if node.is_innermost]

    def get_outermost_loops(self) -> List[NaturalLoop]:
        """Return top-level loops not enclosed in any other loop."""
        return [node.loop for node in self.root.children]

    def nesting_depth(self, loop: NaturalLoop) -> int:
        """Return 1-based nesting depth of loop (1 for top-level loop)."""
        for node in self.all_nodes:
            if node.loop == loop:
                depth = 0
                curr = node
                while curr.parent is not None:
                    depth += 1
                    curr = curr.parent
                return depth
        return 0


# =====================================================================
# Dominator Computation (Cooper-Harvey-Kennedy Algorithm)
# =====================================================================

def compute_dominators(
    cfg: Dict[str, BasicBlock],
) -> Tuple[Dict[str, Optional[str]], Dict[str, Set[str]], Dict[str, List[str]]]:
    """Compute immediate dominators, dominator sets, and dominator tree children.

    Uses the Cooper-Harvey-Kennedy (2001) iterative reverse postorder algorithm.
    Returns:
        idom: Dict[block_label, immediate_dominator_or_None]
        dom: Dict[block_label, Set_of_all_dominators]
        dom_tree: Dict[block_label, List_of_children_in_dominator_tree]
    """
    if not cfg:
        return {}, {}, {}

    entry = next(iter(cfg.keys()))

    # 1. Depth-first search to compute postorder and reverse postorder (RPO)
    visited: Set[str] = set()
    postorder: List[str] = []

    def dfs(u: str):
        visited.add(u)
        if u in cfg:
            for s in cfg[u].succs:
                if s in cfg and s not in visited:
                    dfs(s)
        postorder.append(u)

    dfs(entry)
    rpo = list(reversed(postorder))
    postorder_num = {node: i for i, node in enumerate(postorder)}

    # 2. Cooper-Harvey-Kennedy algorithm
    doms: Dict[str, Optional[str]] = {node: None for node in cfg}
    doms[entry] = entry

    def intersect(b1: str, b2: str) -> str:
        finger1 = b1
        finger2 = b2
        while finger1 != finger2:
            while postorder_num[finger1] < postorder_num[finger2]:
                finger1 = doms[finger1]
            while postorder_num[finger2] < postorder_num[finger1]:
                finger2 = doms[finger2]
        return finger1

    changed = True
    while changed:
        changed = False
        for b in rpo:
            if b == entry:
                continue
            preds = [p for p in cfg[b].preds if p in doms and doms[p] is not None]
            if not preds:
                continue
            new_idom = preds[0]
            for p in preds[1:]:
                new_idom = intersect(p, new_idom)
            if doms[b] != new_idom:
                doms[b] = new_idom
                changed = True

    # Immediate dominators: idom[entry] is None
    idom: Dict[str, Optional[str]] = {b: (doms[b] if b != entry else None) for b in cfg}

    # 3. Full dominator sets dom[b] = {b} U dom[idom[b]]
    dom: Dict[str, Set[str]] = {b: set() for b in cfg}
    for b in rpo:
        curr = b
        while curr is not None:
            dom[b].add(curr)
            curr = idom[curr]

    # 4. Dominator tree children
    dom_tree: Dict[str, List[str]] = {b: [] for b in cfg}
    for b, parent in idom.items():
        if parent is not None and parent in dom_tree:
            dom_tree[parent].append(b)

    return idom, dom, dom_tree


def dominates(a: str, b: str, dom: Dict[str, Set[str]]) -> bool:
    """Returns True if block a dominates block b."""
    return a in dom.get(b, set())


def strictly_dominates(a: str, b: str, dom: Dict[str, Set[str]]) -> bool:
    """Returns True if block a strictly dominates block b (a != b and a dominates b)."""
    return a != b and a in dom.get(b, set())


# =====================================================================
# Natural Loop Detection
# =====================================================================

def find_natural_loops(cfg: Dict[str, BasicBlock]) -> List[NaturalLoop]:
    """Identify natural loops corresponding to backedges in the CFG."""
    idom, dom, _ = compute_dominators(cfg)
    loops: List[NaturalLoop] = []

    for a, block in cfg.items():
        for b in block.succs:
            if dominates(b, a, dom):
                # Back edge: a -> b (header is b, tail is a)
                header = b
                tail = a
                body: Set[str] = {header, tail}
                stack = [tail] if tail != header else []
                while stack:
                    m = stack.pop()
                    if m in cfg:
                        for p in cfg[m].preds:
                            if p in cfg and p not in body:
                                body.add(p)
                                stack.append(p)

                # Identify exit blocks
                exit_blocks: Set[str] = set()
                for node in body:
                    if node in cfg:
                        for s in cfg[node].succs:
                            if s not in body:
                                exit_blocks.add(node)
                                break

                loops.append(
                    NaturalLoop(
                        header=header,
                        body=body,
                        back_edge=(tail, header),
                        exit_blocks=exit_blocks,
                    )
                )

    return loops


def build_loop_nesting_tree(loops: List[NaturalLoop]) -> LoopNestingTree:
    """Construct loop nesting tree from detected natural loops."""
    return LoopNestingTree(loops)


def find_primary_induction_var(
    loop: NaturalLoop, cfg: Dict[str, Any], candidate_var: Optional[str] = None
) -> Optional[Tuple[str, int, str, int, BinOp]]:
    """Detect primary induction variable defined as i = i + c or i = i - c.

    Returns:
        (var_name, step, block_label, instruction_index, instruction) or None
    """
    from ir import IfNz, IfZ

    defs: Dict[str, List[Tuple[str, int, Any]]] = {}
    for node in loop.body:
        if node in cfg:
            for idx, ins in enumerate(cfg[node].instrs):
                dst = _get_def_var(ins)
                if dst:
                    defs.setdefault(dst, []).append((node, idx, ins))

    candidates: List[Tuple[str, int, str, int, BinOp]] = []
    for var, instances in defs.items():
        if len(instances) == 1:
            node, idx, ins = instances[0]
            if isinstance(ins, BinOp) and ins.dst == var:
                if ins.op == "+" and ins.a == var and isinstance(ins.b, int):
                    candidates.append((var, ins.b, node, idx, ins))
                elif ins.op == "+" and ins.b == var and isinstance(ins.a, int):
                    candidates.append((var, ins.a, node, idx, ins))
                elif ins.op == "-" and ins.a == var and isinstance(ins.b, int):
                    candidates.append((var, -ins.b, node, idx, ins))

    if not candidates:
        return None

    if candidate_var:
        for c in candidates:
            if c[0] == candidate_var:
                return c

    # Prioritize candidate that controls loop exit/branch condition
    condition_vars: Set[str] = set()
    for exit_node in loop.exit_blocks:
        if exit_node in cfg:
            for ins in cfg[exit_node].instrs:
                if isinstance(ins, (IfZ, IfNz)):
                    cond_var = ins.cond
                    if isinstance(cond_var, str):
                        condition_vars.add(cond_var)
                        for prev_ins in reversed(cfg[exit_node].instrs):
                            if isinstance(prev_ins, BinOp) and prev_ins.dst == cond_var:
                                if isinstance(prev_ins.a, str):
                                    condition_vars.add(prev_ins.a)
                                if isinstance(prev_ins.b, str):
                                    condition_vars.add(prev_ins.b)

    for c in candidates:
        if c[0] in condition_vars:
            return c

    return candidates[0]

