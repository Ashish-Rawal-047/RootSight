"""Causal graph utilities: d-separation, backdoor, front-door, instruments.

Everything here is graphical, not statistical.  The graph comes from the KPI
semantic contract (analyst-approved edges plus declared unobserved nodes); the
statistical structure screen in `structure.py` can only CONTRADICT it.

d-separation is implemented via the moralised ancestral graph, which is the
standard correct construction:

  1. take the subgraph induced on ancestors(X u Y u Z)
  2. moralise it (join parents that share a child), drop directions
  3. delete Z
  4. X and Y are d-separated given Z iff no path remains between them
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations


class CausalDAG:
    def __init__(self, edges: list[tuple[str, str]], *,
                 unobserved: set[str] | None = None,
                 provenance: dict[tuple[str, str], dict] | None = None):
        self.edges = list(dict.fromkeys(edges))
        self.unobserved = set(unobserved or ())
        self.provenance = dict(provenance or {})
        self.nodes = sorted({n for e in self.edges for n in e} | self.unobserved)
        self._children: dict[str, set[str]] = {n: set() for n in self.nodes}
        self._parents: dict[str, set[str]] = {n: set() for n in self.nodes}
        for a, b in self.edges:
            self._children[a].add(b)
            self._parents[b].add(a)
        if self._has_cycle():
            raise ValueError("declared causal prior is not acyclic")

    # ------------------------------------------------------------------ basics
    def parents(self, n: str) -> set[str]:
        return set(self._parents.get(n, set()))

    def children(self, n: str) -> set[str]:
        return set(self._children.get(n, set()))

    def observed(self) -> set[str]:
        return set(self.nodes) - self.unobserved

    def _has_cycle(self) -> bool:
        colour: dict[str, int] = {}

        def visit(n: str) -> bool:
            colour[n] = 1
            for c in self._children[n]:
                if colour.get(c, 0) == 1:
                    return True
                if colour.get(c, 0) == 0 and visit(c):
                    return True
            colour[n] = 2
            return False

        return any(colour.get(n, 0) == 0 and visit(n) for n in self.nodes)

    def ancestors(self, nodes: set[str]) -> set[str]:
        out, stack = set(nodes), list(nodes)
        while stack:
            n = stack.pop()
            for p in self._parents.get(n, set()):
                if p not in out:
                    out.add(p)
                    stack.append(p)
        return out

    def descendants(self, n: str) -> set[str]:
        out, stack = set(), [n]
        while stack:
            cur = stack.pop()
            for c in self._children.get(cur, set()):
                if c not in out:
                    out.add(c)
                    stack.append(c)
        return out

    def without_outgoing(self, x: str) -> "CausalDAG":
        return self.without_outgoing_set({x})

    def without_outgoing_set(self, xs: set[str]) -> "CausalDAG":
        return CausalDAG([e for e in self.edges if e[0] not in xs],
                         unobserved=self.unobserved, provenance=self.provenance)

    def without_edge(self, a: str, b: str) -> "CausalDAG":
        return CausalDAG([e for e in self.edges if e != (a, b)],
                         unobserved=self.unobserved, provenance=self.provenance)

    # ----------------------------------------------------------- d-separation
    def d_separated(self, x: set[str], y: set[str], z: set[str]) -> bool:
        keep = self.ancestors(set(x) | set(y) | set(z))
        adj: dict[str, set[str]] = {n: set() for n in keep}
        for a, b in self.edges:
            if a in keep and b in keep:
                adj[a].add(b)
                adj[b].add(a)
        # moralise: parents sharing a child become adjacent
        for n in keep:
            ps = [p for p in self._parents.get(n, set()) if p in keep]
            for p, q in combinations(ps, 2):
                adj[p].add(q)
                adj[q].add(p)
        blocked = set(z) & keep
        start = set(x) & keep
        target = set(y) & keep
        if not start or not target:
            return True
        seen, stack = set(start), list(start - blocked)
        while stack:
            n = stack.pop()
            for m in adj[n]:
                if m in blocked or m in seen:
                    continue
                if m in target:
                    return False
                seen.add(m)
                stack.append(m)
        return not (start & target)

    def d_connected(self, x: str, y: str, z: set[str] | None = None) -> bool:
        return not self.d_separated({x}, {y}, set(z or ()))

    # ---------------------------------------------------------------- backdoor
    def satisfies_backdoor(self, x: str, y: str, z: set[str]) -> tuple[bool, str]:
        desc = self.descendants(x) | {x}
        bad = z & desc
        if bad:
            return False, f"adjustment set contains descendant(s) of the treatment: {sorted(bad)}"
        gx = self.without_outgoing(x)
        if not gx.d_separated({x}, {y}, z):
            return False, "at least one backdoor path from treatment to outcome remains open"
        return True, "blocks every backdoor path and contains no descendant of the treatment"

    def backdoor_sets(self, x: str, y: str, *, candidates: set[str] | None = None,
                      max_size: int = 4) -> list[set[str]]:
        pool = sorted((candidates if candidates is not None
                       else set(self.nodes) - {x, y} - self.descendants(x)))
        found: list[set[str]] = []
        for size in range(0, max_size + 1):
            for combo in combinations(pool, size):
                z = set(combo)
                if any(prev <= z for prev in found):     # keep only minimal sets
                    continue
                ok, _ = self.satisfies_backdoor(x, y, z)
                if ok:
                    found.append(z)
        return found

    # -------------------------------------------------------------- front-door
    def directed_paths(self, x: str, y: str, limit: int = 400) -> list[list[str]]:
        paths: list[list[str]] = []

        def walk(n: str, path: list[str]) -> None:
            if len(paths) >= limit:
                return
            for c in sorted(self._children.get(n, set())):
                if c in path:
                    continue
                if c == y:
                    paths.append(path + [c])
                else:
                    walk(c, path + [c])

        walk(x, [x])
        return paths

    def satisfies_frontdoor(self, x: str, y: str, m: set[str]) -> tuple[bool, str]:
        if not m:
            return False, "empty mediator set"
        paths = self.directed_paths(x, y)
        if not paths:
            return False, "no directed path from treatment to outcome"
        for p in paths:
            if not (set(p[1:-1]) & m):
                return False, f"directed path {' -> '.join(p)} is not intercepted by {sorted(m)}"
        # (ii) no unblocked backdoor path treatment -> mediator: check in G with
        #      edges OUT OF the treatment removed
        if not self.without_outgoing(x).d_separated({x}, m, set()):
            return False, "an unblocked backdoor path from treatment to mediator exists"
        # (iii) all backdoor paths mediator -> outcome blocked by the treatment:
        #       check in G with edges OUT OF the mediator set removed
        if not self.without_outgoing_set(m).d_separated(m, {y}, {x}):
            return False, "a backdoor path from mediator to outcome is not blocked by the treatment"
        return True, "intercepts all directed paths, no backdoor to mediator, mediator-outcome backdoors blocked by treatment"

    def frontdoor_sets(self, x: str, y: str, max_size: int = 2) -> list[set[str]]:
        pool = sorted((self.descendants(x) & self.ancestors({y})) - {x, y} - self.unobserved)
        out = []
        for size in range(1, max_size + 1):
            for combo in combinations(pool, size):
                ok, _ = self.satisfies_frontdoor(x, y, set(combo))
                if ok:
                    out.append(set(combo))
        return out

    # ------------------------------------------------------------- instruments
    def instruments(self, x: str, y: str) -> list[str]:
        """Graphical IV: Z d-connected to X, and d-separated from Y once the
        direct causal channel from X is cut."""
        out = []
        gx = self.without_outgoing(x)
        for z in self.observed():
            if z in (x, y) or z in self.descendants(x):
                continue
            if not self.d_connected(z, x):
                continue
            if gx.d_separated({z}, {y}, set()):
                out.append(z)
        return sorted(out)

    # ----------------------------------------------------------------- export
    def as_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "edges": [{"from": a, "to": b, **self.provenance.get((a, b), {})}
                      for a, b in self.edges],
            "unobserved": sorted(self.unobserved),
        }


@dataclass
class AdjustmentResult:
    treatment: str
    outcome: str
    strategy: str                 # BACKDOOR | FRONTDOOR | IV | NONE
    adjustment_set: list[str] = field(default_factory=list)
    all_valid_sets: list[list[str]] = field(default_factory=list)
    uses_unobserved: bool = False
    unobserved_required: list[str] = field(default_factory=list)
    estimand: str = ""
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "treatment": self.treatment, "outcome": self.outcome,
            "strategy": self.strategy, "adjustment_set": self.adjustment_set,
            "all_valid_sets": self.all_valid_sets,
            "uses_unobserved": self.uses_unobserved,
            "unobserved_required": self.unobserved_required,
            "estimand": self.estimand, "reason": self.reason,
        }


def find_identification_strategy(dag: CausalDAG, x: str, y: str) -> AdjustmentResult:
    """Graphical identification search, in the conventional order of preference."""
    observed = dag.observed()

    all_sets = dag.backdoor_sets(x, y)
    obs_sets = [s for s in all_sets if s <= observed]
    if obs_sets:
        best = min(obs_sets, key=lambda s: (len(s), sorted(s)))
        return AdjustmentResult(
            treatment=x, outcome=y, strategy="BACKDOOR",
            adjustment_set=sorted(best),
            all_valid_sets=[sorted(s) for s in obs_sets],
            estimand=(f"E[{y} | do({x})] = sum_z E[{y} | {x}, Z=z] P(Z=z), "
                      f"Z = {sorted(best) or 'empty set'}"),
            reason="a valid backdoor adjustment set exists and is fully observed")

    fd = dag.frontdoor_sets(x, y)
    if fd:
        best = min(fd, key=lambda s: (len(s), sorted(s)))
        return AdjustmentResult(
            treatment=x, outcome=y, strategy="FRONTDOOR", adjustment_set=sorted(best),
            all_valid_sets=[sorted(s) for s in fd],
            estimand=f"front-door formula through M = {sorted(best)}",
            reason="no observed backdoor set exists but a valid front-door mediator set does")

    ivs = dag.instruments(x, y)
    if ivs:
        return AdjustmentResult(
            treatment=x, outcome=y, strategy="IV", adjustment_set=ivs[:1],
            all_valid_sets=[[i] for i in ivs],
            estimand=f"Wald ratio using instrument {ivs[0]}",
            reason="an instrument satisfying the graphical exclusion condition exists")

    needed = sorted({n for s in all_sets for n in s if n in dag.unobserved})
    return AdjustmentResult(
        treatment=x, outcome=y, strategy="NONE", adjustment_set=[],
        all_valid_sets=[sorted(s) for s in all_sets],
        uses_unobserved=bool(needed), unobserved_required=needed,
        estimand="",
        reason=("every valid adjustment set requires unobserved variable(s) "
                f"{needed}; no front-door path and no instrument is available"
                if needed else
                "no valid backdoor set, front-door set or instrument exists in the "
                "declared graph"))


def dag_from_contract() -> CausalDAG:
    from ..contracts.kpi_contract import registry
    reg = registry()
    edges = list(reg.declared_edges())
    unobs = set()
    for u in reg.unobserved_nodes():
        unobs.add(u["node"])
        for target in u["affects"]:
            edges.append((u["node"], target))
    prov = dict(reg.edge_provenance())
    for u in reg.unobserved_nodes():
        for target in u["affects"]:
            prov[(u["node"], target)] = {"rationale": u["reason"],
                                         "approved_by": "declared_unobserved"}
    return CausalDAG(edges, unobserved=unobs, provenance=prov)
