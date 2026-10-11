"""AO* style planning over an acyclic AND/OR route graph with replanning."""

from dataclasses import dataclass
from math import inf


@dataclass(frozen=True)
class Action:
    name: str
    successors: tuple[str, ...]  # All successor subgoals must be solved (AND).
    cost: float


class AOPlanner:
    def __init__(self, actions: dict[str, list[Action]], terminal: set[str], heuristic: dict[str, float]):
        self.actions, self.terminal, self.heuristic = actions, terminal, heuristic
        self.expanded: set[str] = set()

    def value(self, node: str, active: set[str] | None = None) -> float:
        """Evaluate the least-cost solved alternative; reject cycles explicitly."""
        if node in self.terminal:
            return 0.0
        active = set() if active is None else active
        if node in active:
            return inf
        if node in self.expanded:
            # Recompute rather than cache: a road can become blocked between plans.
            pass
        active.add(node)
        options = self.actions.get(node, [])
        if not options:
            result = self.heuristic.get(node, inf)
        else:
            result = min(
                (action.cost + sum(self.value(child, active.copy()) for child in action.successors)
                 for action in options), default=inf
            )
        self.expanded.add(node)
        return result

    def plan(self, start: str) -> tuple[float, list[tuple[str, str]]]:
        value = self.value(start)
        chosen: list[tuple[str, str]] = []
        visited: set[str] = set()

        def trace(node: str) -> None:
            if node in self.terminal or node in visited:
                return
            visited.add(node)
            options = self.actions.get(node, [])
            if not options:
                return
            best = min(options, key=lambda a: a.cost + sum(self.value(c) for c in a.successors))
            chosen.append((node, best.name))
            for child in best.successors:
                trace(child)

        if value < inf:
            trace(start)
        return value, chosen

    def block_action(self, node: str, action_name: str) -> None:
        self.actions[node] = [a for a in self.actions.get(node, []) if a.name != action_name]
        self.expanded.clear()


def demo() -> None:
    actions = {
        "Depot": [Action("expressway", ("NorthHub",), 5), Action("river-road", ("EastHub",), 4)],
        "NorthHub": [Action("deliver by van", ("Customer",), 4)],
        "EastHub": [Action("detour via market", ("Market",), 2)],
        "Market": [Action("last-mile road", ("Customer",), 5)],
    }
    planner = AOPlanner(actions, {"Customer"}, {"Customer": 0})
    cost, plan = planner.plan("Depot")
    print(f"Initial plan, cost={cost:g}: {plan}")
    print("Road event: expressway closed")
    planner.block_action("Depot", "expressway")
    cost, plan = planner.plan("Depot")
    print(f"Replanned route, cost={cost:g}: {plan}")


if __name__ == "__main__":
    demo()
