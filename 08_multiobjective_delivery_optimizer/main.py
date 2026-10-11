"""NSGA-II for a time-windowed delivery tour with distance and lateness costs."""

from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class Customer:
    name: str
    x: float
    y: float
    window_start: float
    window_end: float
    service: float


def distance(a: Customer, b: Customer) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** .5


def objectives(route: tuple[int, ...], depot: Customer, customers: list[Customer]) -> tuple[float, float]:
    current, elapsed, total_distance, lateness = depot, 0.0, 0.0, 0.0
    for idx in route:
        customer = customers[idx]
        leg = distance(current, customer)
        elapsed += leg
        lateness += max(0.0, elapsed - customer.window_end)
        elapsed = max(elapsed, customer.window_start) + customer.service
        total_distance += leg
        current = customer
    total_distance += distance(current, depot)
    return total_distance, lateness


def dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def rank_population(population: list[tuple[int, ...]], scores: list[tuple[float, float]]) -> tuple[list[int], list[float]]:
    dominates_list = [[] for _ in population]
    dominated_count = [0] * len(population)
    fronts: list[list[int]] = [[]]
    for i, a in enumerate(scores):
        for j, b in enumerate(scores):
            if i == j:
                continue
            if dominates(a, b):
                dominates_list[i].append(j)
            elif dominates(b, a):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            fronts[0].append(i)
    ranks = [0] * len(population)
    for level, front in enumerate(fronts):
        next_front = []
        for i in front:
            ranks[i] = level
            for j in dominates_list[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
    crowding = [0.0] * len(population)
    for front in fronts:
        for objective in range(2):
            ordered = sorted(front, key=lambda i: scores[i][objective])
            crowding[ordered[0]] = crowding[ordered[-1]] = float("inf")
            span = scores[ordered[-1]][objective] - scores[ordered[0]][objective]
            if span:
                for k in range(1, len(ordered) - 1):
                    crowding[ordered[k]] += (scores[ordered[k + 1]][objective] - scores[ordered[k - 1]][objective]) / span
    return ranks, crowding


def two_opt(route: tuple[int, ...], depot: Customer, customers: list[Customer]) -> tuple[int, ...]:
    best = route
    improved = True
    while improved:
        improved = False
        base = objectives(best, depot, customers)
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + tuple(reversed(best[i:j + 1])) + best[j + 1:]
                score = objectives(candidate, depot, customers)
                if score[0] <= base[0] and score[1] <= base[1] and score != base:
                    best, improved = candidate, True
                    break
            if improved:
                break
    return best


def optimize(depot: Customer, customers: list[Customer], population_size: int = 80, generations: int = 100, seed: int = 7) -> list[tuple[tuple[int, ...], tuple[float, float]]]:
    rng = Random(seed)
    n = len(customers)
    population = [tuple(rng.sample(range(n), n)) for _ in range(population_size)]
    for generation in range(generations):
        scores = [objectives(route, depot, customers) for route in population]
        ranks, crowding = rank_population(population, scores)

        def tournament() -> tuple[int, ...]:
            a, b = rng.randrange(len(population)), rng.randrange(len(population))
            return population[a if (ranks[a], -crowding[a]) < (ranks[b], -crowding[b]) else b]

        offspring = []
        while len(offspring) < population_size:
            p1, p2 = tournament(), tournament()
            left, right = sorted(rng.sample(range(n), 2)) if n > 1 else (0, 0)
            child = [None] * n
            child[left:right] = p1[left:right]
            fill = [gene for gene in p2 if gene not in child]
            cursor = 0
            for i in range(n):
                if child[i] is None:
                    child[i] = fill[cursor]
                    cursor += 1
            if rng.random() < .25 and n > 1:
                i, j = rng.sample(range(n), 2)
                child[i], child[j] = child[j], child[i]
            offspring.append(two_opt(tuple(child), depot, customers))
        combined = population + offspring
        combined_scores = [objectives(route, depot, customers) for route in combined]
        ranks2, crowd2 = rank_population(combined, combined_scores)
        order = sorted(range(len(combined)), key=lambda i: (ranks2[i], -crowd2[i]))
        population = [combined[i] for i in order[:population_size]]
        if generation % 25 == 0:
            print(f"generation {generation:3}: population={len(population)}")
    scores = [objectives(route, depot, customers) for route in population]
    ranks, _ = rank_population(population, scores)
    return [(route, score) for route, score, rank in zip(population, scores, ranks) if rank == 0]


def demo() -> None:
    depot = Customer("Depot", 0, 0, 0, 100, 0)
    customers = [
        Customer("A", 2, 8, 0, 11, 1), Customer("B", 5, 2, 3, 12, 1),
        Customer("C", 8, 7, 0, 15, 1), Customer("D", 1, 3, 2, 8, 1),
        Customer("E", 7, 1, 5, 14, 1), Customer("F", 4, 6, 4, 13, 1),
        Customer("G", 9, 4, 2, 12, 1),
    ]
    front = optimize(depot, customers)
    print("\nNon-dominated delivery plans:")
    for route, (km, late) in sorted(front, key=lambda item: item[1][0])[:10]:
        names = [customers[i].name for i in route]
        print(f"  {names}: {km:.2f} distance units, {late:.2f} total late units")


if __name__ == "__main__":
    demo()
