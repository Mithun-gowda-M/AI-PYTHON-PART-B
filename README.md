# AI Python Programs — Part B

Eight advanced, self-contained projects inspired by the open-ended experiments in the AI Laboratory Manual. Six implement the manual's topics; projects 7 and 8 extend them with multi-agent planning and multi-objective optimization. Each project lives in its own folder and uses only Python's standard library (Python 3.10+).

## Projects

| Folder | Project | Main ideas |
|---|---|---|
| `01_ao_star_dynamic_routes` | Dynamic AND/OR route planner | AO* style cost propagation, compound actions, blocked-road replanning |
| `02_adversarial_connect_four` | Adversarial game-playing agent | Iterative-deepening minimax, alpha-beta pruning, move ordering, transposition table |
| `03_first_order_logic_agent` | First-order logic reasoning agent | Unification, forward chaining, variable renaming, proof provenance |
| `04_bayesian_decision_advisor` | Decisions under uncertainty | Bayesian updates, likelihood ratios, expected utility, value of information |
| `05_bayesian_network_engine` | Bayesian network inference | DAG validation, factor algebra, variable elimination, evidence conditioning |
| `06_explainable_expert_system` | Explainable expert system | Certainty factors, rule conflict handling, evidence and explanation traces |
| `07_multi_robot_cbs` | Cooperative warehouse robots | Conflict-Based Search, space-time A*, vertex and edge-swap conflicts |
| `08_multiobjective_delivery_optimizer` | Multi-objective delivery planner | NSGA-II, Pareto fronts, crowding distance, time windows and local search |

Run a project from its folder, for example:

```bash
python 01_ao_star_dynamic_routes/main.py
```

Run the regression checks from the repository root:

```bash
python -m unittest discover -s tests -v
```

These are educational prototypes. Their models, probabilities, utilities, and constraints are illustrative and should be replaced with validated domain data for real deployments.
