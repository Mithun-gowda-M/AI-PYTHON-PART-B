"""Bayesian posterior updates, expected-utility action selection, and VOI."""

from math import prod


def normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Evidence has zero probability under every hypothesis")
    return {key: value / total for key, value in weights.items()}


def posterior(priors: dict[str, float], evidence_likelihoods: list[dict[str, float]]) -> dict[str, float]:
    if abs(sum(priors.values()) - 1.0) > 1e-9:
        raise ValueError("Priors must sum to 1")
    return normalize({h: p * prod(item[h] for item in evidence_likelihoods) for h, p in priors.items()})


def expected_utilities(belief: dict[str, float], utilities: dict[str, dict[str, float]]) -> dict[str, float]:
    return {action: sum(prob * utilities[action][hypothesis] for hypothesis, prob in belief.items())
            for action in utilities}


def value_of_information(
    belief: dict[str, float],
    test: dict[str, dict[str, float]],
    utilities: dict[str, dict[str, float]],
    test_cost: float,
) -> float:
    """Expected gain from observing test outcomes before choosing an action."""
    current_best = max(expected_utilities(belief, utilities).values())
    expected_after = 0.0
    outcomes = next(iter(test.values())).keys()
    for outcome in outcomes:
        likelihood = {h: test[h][outcome] for h in belief}
        probability = sum(belief[h] * likelihood[h] for h in belief)
        if probability == 0:
            continue
        updated = normalize({h: belief[h] * likelihood[h] for h in belief})
        expected_after += probability * max(expected_utilities(updated, utilities).values())
    return expected_after - current_best - test_cost


def demo() -> None:
    priors = {"viral": 0.60, "bacterial": 0.25, "other": 0.15}
    evidence = [
        {"viral": 0.75, "bacterial": 0.55, "other": 0.30},  # fever
        {"viral": 0.65, "bacterial": 0.70, "other": 0.20},  # cough
        {"viral": 0.10, "bacterial": 0.80, "other": 0.15},  # high inflammatory marker
    ]
    belief = posterior(priors, evidence)
    actions = {
        "supportive_care": {"viral": 8, "bacterial": 2, "other": 3},
        "clinician_review": {"viral": 5, "bacterial": 6, "other": 5},
        "immediate_escalation": {"viral": -4, "bacterial": 8, "other": 0},
    }
    print("Posterior belief:")
    for hypothesis, probability in belief.items():
        print(f"  {hypothesis:12} {probability:.3f}")
    utilities = expected_utilities(belief, actions)
    print("Expected utility:", utilities)
    test = {h: {"positive": p, "negative": 1 - p} for h, p in
            {"viral": 0.15, "bacterial": 0.85, "other": 0.25}.items()}
    print("Net value of additional culture test:", f"{value_of_information(belief, test, actions, test_cost=0.4):.3f}")
    print("Educational decision model only; probabilities and utilities are illustrative.")


if __name__ == "__main__":
    demo()
