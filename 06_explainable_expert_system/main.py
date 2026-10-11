"""Rule-based expert system with certainty-factor aggregation and explanations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    name: str
    premises: frozenset[str]
    conclusion: str
    strength: float
    recommendation: str


def combine_cf(a: float, b: float) -> float:
    """MYCIN-style combination for independent evidence with same sign."""
    if not -1 <= a <= 1 or not -1 <= b <= 1:
        raise ValueError("Certainty factors must be between -1 and 1")
    if a >= 0 and b >= 0:
        return a + b * (1 - a)
    if a <= 0 and b <= 0:
        return a + b * (1 + a)
    return (a + b) / (1 - min(abs(a), abs(b)))


def infer(observations: dict[str, float], rules: list[Rule]) -> tuple[dict[str, float], dict[str, list[str]]]:
    beliefs = dict(observations)
    traces: dict[str, list[str]] = {fact: [f"Observed {fact} at CF={cf:.2f}"] for fact, cf in observations.items()}
    fired: set[str] = set()
    for _ in range(len(rules) + 1):
        changed = False
        for rule in rules:
            if rule.name in fired or not all(p in beliefs and beliefs[p] > 0 for p in rule.premises):
                continue
            premise_cf = min(beliefs[p] for p in rule.premises)
            support = premise_cf * rule.strength
            previous = beliefs.get(rule.conclusion, 0.0)
            updated = combine_cf(previous, support)
            beliefs[rule.conclusion] = updated
            traces.setdefault(rule.conclusion, []).append(
                f"{rule.name}: {sorted(rule.premises)} (CF={premise_cf:.2f}) x strength {rule.strength:.2f} => +{support:.2f}; combined={updated:.2f}"
            )
            fired.add(rule.name)
            changed = True
        if not changed:
            break
    return beliefs, traces


def demo() -> None:
    rules = [
        Rule("R1", frozenset({"difficulty_starting", "dim_lights"}), "battery_fault", .90, "Inspect battery and charging system"),
        Rule("R2", frozenset({"battery_fault", "corroded_terminals"}), "battery_service_needed", .95, "Clean terminals and test battery"),
        Rule("R3", frozenset({"high_temperature", "steam"}), "overheating", .92, "Stop safely; inspect coolant and radiator"),
        Rule("R4", frozenset({"soft_brake", "squeaking"}), "brake_fault", .95, "Avoid driving; arrange brake inspection"),
        Rule("R5", frozenset({"warning_light", "difficulty_starting"}), "electrical_fault", .55, "Check diagnostic codes"),
    ]
    observations = {"difficulty_starting": .95, "dim_lights": .80, "warning_light": .65, "corroded_terminals": .90}
    beliefs, traces = infer(observations, rules)
    print("Ranked hypotheses:")
    for hypothesis, confidence in sorted(beliefs.items(), key=lambda item: item[1], reverse=True):
        if hypothesis not in observations:
            print(f"\n{hypothesis}: CF={confidence:.3f}")
            for line in traces[hypothesis]:
                print("  ", line)
    print("\nRecommendations:")
    for rule in rules:
        if rule.conclusion in beliefs and beliefs[rule.conclusion] >= .50:
            print(f"  {rule.conclusion}: {rule.recommendation} (CF={beliefs[rule.conclusion]:.2f})")


if __name__ == "__main__":
    demo()
