"""Discrete Bayesian network inference by variable elimination."""

from dataclasses import dataclass
from itertools import product


@dataclass
class Factor:
    variables: tuple[str, ...]
    table: dict[tuple[str, ...], float]

    def restrict(self, evidence: dict[str, str]) -> "Factor":
        kept = tuple(v for v in self.variables if v not in evidence)
        table = {}
        for assignment, value in self.table.items():
            mapping = dict(zip(self.variables, assignment))
            if all(mapping.get(v) == state for v, state in evidence.items() if v in mapping):
                table[tuple(mapping[v] for v in kept)] = value
        return Factor(kept, table)

    def sum_out(self, variable: str) -> "Factor":
        if variable not in self.variables:
            return self
        kept = tuple(v for v in self.variables if v != variable)
        table: dict[tuple[str, ...], float] = {}
        for assignment, value in self.table.items():
            reduced = tuple(state for v, state in zip(self.variables, assignment) if v != variable)
            table[reduced] = table.get(reduced, 0.0) + value
        return Factor(kept, table)

    def multiply(self, other: "Factor") -> "Factor":
        union = self.variables + tuple(v for v in other.variables if v not in self.variables)
        result: dict[tuple[str, ...], float] = {}
        left_index = [union.index(v) for v in self.variables]
        right_index = [union.index(v) for v in other.variables]
        for a, va in self.table.items():
            for b, vb in other.table.items():
                merged = [None] * len(union)
                for i, state in zip(left_index, a):
                    merged[i] = state
                if any(merged[i] is not None and merged[i] != state for i, state in zip(right_index, b)):
                    continue
                for i, state in zip(right_index, b):
                    merged[i] = state
                result[tuple(merged)] = va * vb
        return Factor(union, result)

    def normalize(self) -> "Factor":
        total = sum(self.table.values())
        if total == 0:
            raise ValueError("Evidence is impossible under this network")
        return Factor(self.variables, {k: v / total for k, v in self.table.items()})


class BayesianNetwork:
    def __init__(self, domains: dict[str, tuple[str, ...]], factors: list[Factor]):
        self.domains, self.factors = domains, factors
        self.validate()

    def validate(self) -> None:
        for factor in self.factors:
            if any(v not in self.domains for v in factor.variables):
                raise ValueError("Factor references an unknown variable")
            expected = set(product(*(self.domains[v] for v in factor.variables)))
            if set(factor.table) != expected:
                raise ValueError(f"Factor {factor.variables} has incomplete assignments")
            if any(p < 0 for p in factor.table.values()):
                raise ValueError("Probabilities cannot be negative")
        # A variable elimination model must be a normalized joint factorization.
        for factor in self.factors:
            if factor.variables:
                child, parents = factor.variables[0], factor.variables[1:]
                parent_assignments = product(*(self.domains[v] for v in parents))
                for parent_states in parent_assignments:
                    row_total = sum(
                        factor.table[(child_state, *parent_states)]
                        for child_state in self.domains[child]
                    )
                    if abs(row_total - 1.0) > 1e-8:
                        raise ValueError(f"Conditional probabilities for {child} and {parents} must sum to 1")

    def min_fill_order(self, query: str, evidence: set[str]) -> list[str]:
        scopes = [set(f.variables) for f in self.factors]
        candidates = set(self.domains) - evidence - {query}
        order = []
        while candidates:
            def fill_cost(v: str) -> tuple[int, str]:
                neighbors = set().union(*(s - {v} for s in scopes if v in s))
                missing = sum(1 for a in neighbors for b in neighbors if a < b and not any({a, b} <= s for s in scopes))
                return missing, v
            variable = min(candidates, key=fill_cost)
            related = [s for s in scopes if variable in s]
            merged = set().union(*related) if related else {variable}
            scopes = [s for s in scopes if variable not in s] + [merged - {variable}]
            order.append(variable)
            candidates.remove(variable)
        return order

    def query(self, target: str, evidence: dict[str, str]) -> Factor:
        if target not in self.domains or any(v not in self.domains or s not in self.domains[v] for v, s in evidence.items()):
            raise ValueError("Unknown query/evidence variable or state")
        if target in evidence:
            return Factor((target,), {(state,): float(state == evidence[target]) for state in self.domains[target]})
        factors = [f.restrict(evidence) for f in self.factors]
        for variable in self.min_fill_order(target, set(evidence)):
            related = [f for f in factors if variable in f.variables]
            factors = [f for f in factors if variable not in f.variables]
            if related:
                combined = related[0]
                for factor in related[1:]:
                    combined = combined.multiply(factor)
                factors.append(combined.sum_out(variable))
        combined = factors[0]
        for factor in factors[1:]:
            combined = combined.multiply(factor)
        for variable in tuple(v for v in combined.variables if v != target):
            combined = combined.sum_out(variable)
        return combined.normalize()


def demo() -> None:
    domains = {"Rain": ("yes", "no"), "Sprinkler": ("on", "off"), "WetGrass": ("yes", "no")}
    factors = [
        Factor(("Rain",), {("yes",): .2, ("no",): .8}),
        Factor(("Sprinkler", "Rain"), {("on", "yes"): .01, ("off", "yes"): .99, ("on", "no"): .4, ("off", "no"): .6}),
        Factor(("WetGrass", "Rain", "Sprinkler"), {
            ("yes", "yes", "on"): .99, ("no", "yes", "on"): .01,
            ("yes", "yes", "off"): .8, ("no", "yes", "off"): .2,
            ("yes", "no", "on"): .9, ("no", "no", "on"): .1,
            ("yes", "no", "off"): .0, ("no", "no", "off"): 1.0,
        }),
    ]
    network = BayesianNetwork(domains, factors)
    result = network.query("Rain", {"WetGrass": "yes"})
    print("P(Rain | WetGrass=yes):")
    for state in domains["Rain"]:
        print(f"  {state:3}: {result.table[(state,)]:.4f}")
    print("Elimination order:", network.min_fill_order("Rain", {"WetGrass"}))


if __name__ == "__main__":
    demo()
