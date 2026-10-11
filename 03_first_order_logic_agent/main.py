"""First-order Horn-clause inference with unification and proof provenance."""

from dataclasses import dataclass
from itertools import count


@dataclass(frozen=True)
class Atom:
    predicate: str
    terms: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.predicate}({', '.join(self.terms)})"


@dataclass(frozen=True)
class Rule:
    premises: tuple[Atom, ...]
    conclusion: Atom
    label: str


def is_var(term: str) -> bool:
    return term.startswith("?")


def deref(term: str, subst: dict[str, str]) -> str:
    while is_var(term) and term in subst:
        term = subst[term]
    return term


def unify(a: Atom, b: Atom, subst: dict[str, str] | None = None) -> dict[str, str] | None:
    if a.predicate != b.predicate or len(a.terms) != len(b.terms):
        return None
    result = dict(subst or {})
    for left, right in zip(a.terms, b.terms):
        left, right = deref(left, result), deref(right, result)
        if left == right:
            continue
        if is_var(left):
            result[left] = right
        elif is_var(right):
            result[right] = left
        else:
            return None
    return result


def substitute(atom: Atom, subst: dict[str, str]) -> Atom:
    return Atom(atom.predicate, tuple(deref(t, subst) for t in atom.terms))


def standardize(rule: Rule, serial: int) -> Rule:
    variables = {t for atom in (*rule.premises, rule.conclusion) for t in atom.terms if is_var(t)}
    renaming = {v: f"?{serial}_{v[1:]}" for v in variables}
    return Rule(tuple(substitute(a, renaming) for a in rule.premises), substitute(rule.conclusion, renaming), rule.label)


def infer(facts: list[Atom], rules: list[Rule], max_rounds: int = 100) -> tuple[dict[Atom, str], list[str]]:
    proofs = {fact: f"Given fact: {fact}" for fact in facts}
    serial, trace = count(1), []
    for _ in range(max_rounds):
        additions: dict[Atom, str] = {}
        for original in rules:
            rule = standardize(original, next(serial))

            def join(index: int, subst: dict[str, str], used: list[Atom]) -> None:
                if index == len(rule.premises):
                    result = substitute(rule.conclusion, subst)
                    if result not in proofs and result not in additions:
                        premises = " AND ".join(map(str, used))
                        additions[result] = f"{rule.label}: {premises} => {result}"
                    return
                goal = substitute(rule.premises[index], subst)
                for known in tuple(proofs):
                    next_subst = unify(goal, known, subst)
                    if next_subst is not None:
                        join(index + 1, next_subst, used + [known])

            join(0, {}, [])
        if not additions:
            return proofs, trace
        proofs.update(additions)
        trace.extend(additions.values())
    raise RuntimeError("Inference did not converge; check for a non-finite rule set")


def explain(atom: Atom, proofs: dict[Atom, str], seen: set[Atom] | None = None, indent: int = 0) -> None:
    seen = set() if seen is None else seen
    print(" " * indent + proofs.get(atom, f"No proof for {atom}"))
    if atom in seen:
        return
    seen.add(atom)
    # Provenance strings are human-readable; the trace below is the compact derivation.


def demo() -> None:
    facts = [Atom("HasSymptom", ("Mira", "fever")), Atom("HasSymptom", ("Mira", "cough")), Atom("HasSymptom", ("Mira", "sore_throat"))]
    rules = [
        Rule((Atom("HasSymptom", ("?p", "fever")), Atom("HasSymptom", ("?p", "cough")), Atom("HasSymptom", ("?p", "sore_throat"))), Atom("PossibleCondition", ("?p", "flu")), "R1"),
        Rule((Atom("PossibleCondition", ("?p", "flu")), Atom("HasSymptom", ("?p", "fatigue"))), Atom("Recommend", ("?p", "rest_and_hydration")), "R2"),
    ]
    proofs, trace = infer(facts, rules)
    query = Atom("PossibleCondition", ("Mira", "flu"))
    print("Query:", query, "=>", "PROVED" if query in proofs else "NOT PROVED")
    print("Forward-chaining trace:")
    for step in trace:
        print(" -", step)
    if query in proofs:
        print("Proof leaf:", proofs[query])


if __name__ == "__main__":
    demo()
