#!/usr/bin/env python3
"""
Optimal Set of Bucket Orders Problem (OSBOP) Matheuristic.

This script uses Gurobi to solve the OSBOP by parsing a custom `.dat` instance
and applying a multi-start alternating-optimization matheuristic. Each
iteration successively solves a bucket-order-update MILP and a weight-update
LP. The script writes both a detailed solution report and a summary of the
execution.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Default input, output, summary, and machine-readable warm-start locations.
DEFAULT_DATAFILE = BASE_DIR / "../inputs/2/R1.dat"
DEFAULT_OUTFILE = BASE_DIR / "../outputs/2/solutionsOSBOP_heu.txt"
DEFAULT_SUMMARYFILE = BASE_DIR / "../outputs/2/summaryOSBOP_heu.txt"
DEFAULT_WARMSTART_DIR = BASE_DIR / "../outputs/2/warmstarts/OSBOP_heu"

# Default matheuristic and optimization parameters.
DEFAULT_NUM_COMPONENTS = 2
DEFAULT_TOTAL_TIME_LIMIT = 20000.0
DEFAULT_PHASE_1_TIME_LIMIT = 120.0
DEFAULT_MAX_ITERATIONS = 6
DEFAULT_STARTS = 10
DEFAULT_SEED = 123
DEFAULT_TOLERANCE = 1e-5


# ---------------------------------------------------------------------------
# Instance parser
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstanceData:
    """Data read from one OSBOP instance file."""

    n: int
    a: List[List[float]]
    dispersion: Optional[int] = None


def parse_scalar_int(text: str, key: str) -> Optional[int]:
    """Read an optional integer field from a custom .dat file."""
    match = re.search(
        rf"\b{re.escape(key)}\s*:\s*(-?\d+)\b",
        text,
        flags=re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def parse_matrix_a(text: str, n: int) -> List[List[float]]:
    """Read matrix a, including integer and half-integer values."""
    match = re.search(
        r"\ba\s*:\s*\[\s*(.*?)\s*\]",
        text,
        flags=re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    if not match:
        raise ValueError("Cannot find block 'a: [ ... ]'.")

    number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    values = [float(value) for value in re.findall(number_pattern, match.group(1))]

    expected_size = n * n
    if len(values) != expected_size:
        raise ValueError(
            f"Incorrect size for matrix a: expected {expected_size}, "
            f"found {len(values)}."
        )

    a = [[0.0] * (n + 1) for _ in range(n + 1)]
    position = 0
    for r in range(1, n + 1):
        for s in range(1, n + 1):
            a[r][s] = values[position]
            position += 1
    return a


def load_instance(path: Path) -> InstanceData:
    """Load either a historical OSBOP instance or a synthetic instance."""
    text = path.read_text(encoding="utf-8", errors="strict")
    n = parse_scalar_int(text, "n")
    if n is None:
        raise ValueError("Cannot find n in the instance file.")

    return InstanceData(
        n=n,
        a=parse_matrix_a(text, n),
        dispersion=parse_scalar_int(text, "D"),
    )


def build_normalized_matrix(instance: InstanceData) -> List[List[float]]:
    """Normalize pairwise counts into a pair order matrix C."""
    n = instance.n
    c = [[0.0] * (n + 1) for _ in range(n + 1)]

    for r in range(1, n + 1):
        c[r][r] = 0.5
        for s in range(r + 1, n + 1):
            denominator = instance.a[r][s] + instance.a[s][r]
            if denominator < -1e-12:
                raise ValueError(f"Negative pairwise mass for pair ({r},{s}).")

            if abs(denominator) <= 1e-12:
                c[r][s] = 0.5
                c[s][r] = 0.5
            else:
                c[r][s] = instance.a[r][s] / denominator
                c[s][r] = instance.a[s][r] / denominator

            if not (-1e-9 <= c[r][s] <= 1 + 1e-9):
                raise ValueError(f"Invalid normalized value for pair ({r},{s}).")

    return c


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

XValues = Dict[Tuple[int, int, int], int]


@dataclass(frozen=True)
class SolutionState:
    """Serializable feasible solution used to initialize the next value of g."""

    n: int
    g: int
    objective: float
    weights: List[float]
    x_values: XValues


def state_path(warmstart_dir: Path, datafile: Path, g: int) -> Path:
    """Return the unique machine-readable solution path for one run."""
    return warmstart_dir / f"{datafile.stem}_g{g}.json"


def buckets_to_x_values(
    n: int,
    component_buckets: Sequence[Sequence[Sequence[int]]],
) -> XValues:
    """Convert bucket orders into directed binary relations."""
    expected_items = set(range(1, n + 1))
    x_values: XValues = {}

    for component, buckets in enumerate(component_buckets, start=1):
        positions: Dict[int, int] = {}
        for position, bucket in enumerate(buckets):
            for item in bucket:
                item = int(item)
                if item in positions:
                    raise ValueError(
                        f"Item {item} appears more than once in component {component}."
                    )
                positions[item] = position

        if set(positions) != expected_items:
            raise ValueError(
                f"Component {component} does not contain every item exactly once."
            )

        for r in expected_items:
            for s in expected_items:
                if r != s:
                    x_values[(r, s, component)] = (
                        1 if positions[r] <= positions[s] else 0
                    )

    return x_values


def load_inherited_state(
    path: Path,
    n: int,
    target_g: int,
    c: List[List[float]],
) -> SolutionState:
    """Load the solution for g-1 and append zero-weight tied components."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_n = int(payload["n"])
    source_g = int(payload["g"])

    if source_n != n:
        raise ValueError(
            f"Warm start {path} has n={source_n}, but the instance has n={n}."
        )
    if source_g != target_g - 1:
        raise ValueError(
            f"Warm start {path} has g={source_g}; expected g={target_g - 1}."
        )

    raw_weights = [float(value) for value in payload["weights"]]
    raw_components = payload["components"]
    if len(raw_weights) != source_g or len(raw_components) != source_g:
        raise ValueError(f"Incomplete warm-start data in {path}.")

    weights = [0.0] + raw_weights + [0.0] * (target_g - source_g)
    components = list(raw_components)
    for _ in range(source_g + 1, target_g + 1):
        components.append([list(range(1, n + 1))])

    if any(weights[i] + 1e-9 < weights[i + 1] for i in range(1, target_g)):
        raise ValueError(f"Warm-start weights in {path} are not nonincreasing.")
    if abs(sum(weights[1:]) - 1.0) > 1e-6:
        raise ValueError(f"Warm-start weights in {path} do not sum to one.")

    x_values = buckets_to_x_values(n, components)
    objective = evaluate_solution(n, target_g, c, x_values, weights)
    return SolutionState(n, target_g, objective, weights, x_values)


def save_solution_state(path: Path, state: SolutionState) -> None:
    """Atomically save the best solution for use by the run with g+1."""
    components: List[List[List[int]]] = []
    for component in range(1, state.g + 1):
        buckets, _ = extract_bucket_order(state.n, state.x_values, component)
        components.append(buckets)

    payload = {
        "n": state.n,
        "g": state.g,
        "objective": state.objective,
        "weights": state.weights[1:],
        "components": components,
    }

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def set_threads_from_slurm(model: gp.Model) -> None:
    """Use the number of threads assigned to the SLURM job."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus and slurm_cpus.isdigit():
        model.setParam("Threads", int(slurm_cpus))


def fmt_real(value: float, max_decimals: int = 10) -> str:
    """Format a real value without unnecessary trailing zeroes."""
    text = f"{value:.{max_decimals}f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def optional_int(value: Optional[int]) -> str:
    """Format optional metadata in a summary table."""
    return "-" if value is None else str(value)


def append_text_locked(
    path: Path,
    text: str,
    header: Optional[str] = None,
) -> None:
    """Append a complete block safely to a shared experiment file."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a+", encoding="utf-8", newline="\n") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        try:
            file.seek(0, os.SEEK_END)
            if header is not None and file.tell() == 0:
                file.write(header)
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        finally:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def make_wls_env(max_retries: int = 3) -> gp.Env:
    """Create a WLS environment, retrying transient token failures."""
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        environment = gp.Env(empty=True)
        environment.setParam("WLSTokenDuration", 60)
        environment.setParam("WLSTokenRefresh", 1)

        try:
            environment.start()
            return environment
        except gp.GurobiError as error:
            last_error = error
            try:
                environment.dispose()
            except Exception:
                pass
            time.sleep(2.0 * attempt + random.random())

    if last_error is not None:
        raise last_error
    raise RuntimeError("Could not start the WLS environment.")


# ---------------------------------------------------------------------------
# Weak orders and objective evaluation
# ---------------------------------------------------------------------------

def evaluate_solution(
    n: int,
    g: int,
    c: List[List[float]],
    x_values: XValues,
    weights: Sequence[float],
) -> float:
    """Evaluate the upper-triangular OSBOP objective."""
    objective = 0.0

    for r in range(1, n + 1):
        for s in range(r + 1, n + 1):
            fitted_difference = sum(
                weights[component]
                * (
                    x_values[(r, s, component)]
                    - x_values[(s, r, component)]
                )
                for component in range(1, g + 1)
            )
            observed_difference = c[r][s] - c[s][r]
            objective += abs(fitted_difference - observed_difference)

    return objective


def extract_bucket_order(
    n: int,
    x_values: XValues,
    component: int,
) -> Tuple[List[List[int]], Dict[int, int]]:
    """Extract ordered buckets using exact doubled weak-order scores."""
    items = range(1, n + 1)
    score2: Dict[int, int] = {item: 0 for item in items}

    for r in items:
        for s in items:
            if r == s:
                continue

            x_rs = x_values[(r, s, component)]
            x_sr = x_values[(s, r, component)]
            if x_rs == 1 and x_sr == 1:
                score2[r] += 1
            elif x_rs == 1 and x_sr == 0:
                score2[r] += 2
            elif x_rs == 0 and x_sr == 1:
                pass
            else:
                raise RuntimeError(
                    f"Incomplete relation between items {r} and {s} "
                    f"in component {component}."
                )

    buckets_by_score: Dict[int, List[int]] = {}
    for item, score in score2.items():
        buckets_by_score.setdefault(score, []).append(item)

    buckets = [
        sorted(buckets_by_score[score])
        for score in sorted(buckets_by_score, reverse=True)
    ]
    return buckets, score2


def format_bucket_order(buckets: Sequence[Sequence[int]]) -> str:
    """Format buckets using '|' as the strict-precedence separator."""
    return " | ".join(" ".join(str(item) for item in bucket) for bucket in buckets)


def validate_bucket_order(
    n: int,
    x_values: XValues,
    component: int,
    buckets: Sequence[Sequence[int]],
) -> None:
    """Check that extracted buckets reproduce all binary relations."""
    positions = {
        item: position
        for position, bucket in enumerate(buckets)
        for item in bucket
    }

    for r in range(1, n + 1):
        for s in range(1, n + 1):
            if r == s:
                continue
            expected = 1 if positions[r] <= positions[s] else 0
            if x_values[(r, s, component)] != expected:
                raise RuntimeError(
                    "The extracted bucket order does not reproduce the "
                    f"relation x[{r},{s},{component}]."
                )


# ---------------------------------------------------------------------------
# Phase 1: fixed weights, optimize bucket orders
# ---------------------------------------------------------------------------

def phase_1(
    n: int,
    g: int,
    c: List[List[float]],
    weights: Sequence[float],
    x_start: Optional[XValues],
    verbose: int,
    time_limit: float,
    environment: gp.Env,
) -> Tuple[Optional[XValues], Optional[float]]:
    """Solve the ranking-update MILP for fixed component weights."""
    items = tuple(range(1, n + 1))
    groups = tuple(range(1, g + 1))
    model = gp.Model("OSBOP_PHASE_1", env=environment)

    try:
        model.setParam("MIPGap", DEFAULT_TOLERANCE)
        model.setParam("OutputFlag", 1 if verbose else 0)
        model.setParam("LogToConsole", 1 if verbose else 0)
        set_threads_from_slurm(model)
        if time_limit > 0:
            model.setParam("TimeLimit", float(time_limit))

        x = {
            (r, s, component): model.addVar(
                vtype=GRB.BINARY,
                name=f"x_{r}_{s}_{component}",
            )
            for r in items
            for s in items
            for component in groups
            if r != s
        }
        deviations = {
            (r, s): model.addVar(lb=0.0, name=f"v_{r}_{s}")
            for r in items
            for s in items
            if r < s
        }

        model.setObjective(
            gp.quicksum(deviations[(r, s)] for r in items for s in items if r < s),
            GRB.MINIMIZE,
        )

        for component in groups:
            for r in items:
                for s in items:
                    if r < s:
                        model.addConstr(
                            x[(r, s, component)] + x[(s, r, component)] >= 1
                        )

        for component in groups:
            for r in items:
                for s in items:
                    if s == r:
                        continue
                    for t in items:
                        if t == r or t == s:
                            continue
                        model.addConstr(
                            x[(r, s, component)] + x[(s, t, component)]
                            <= 1 + x[(r, t, component)]
                        )

        for r in items:
            for s in items:
                if r >= s:
                    continue

                fitted_difference = gp.quicksum(
                    weights[component]
                    * (
                        x[(r, s, component)]
                        - x[(s, r, component)]
                    )
                    for component in groups
                )
                observed_difference = c[r][s] - c[s][r]
                model.addConstr(
                    deviations[(r, s)]
                    >= fitted_difference - observed_difference
                )
                model.addConstr(
                    deviations[(r, s)]
                    >= observed_difference - fitted_difference
                )

        if x_start is not None:
            for key, value in x_start.items():
                if key in x:
                    x[key].Start = float(value)

            for r in items:
                for s in items:
                    if r >= s:
                        continue
                    fitted_start = sum(
                        weights[component]
                        * (
                            x_start[(r, s, component)]
                            - x_start[(s, r, component)]
                        )
                        for component in groups
                    )
                    observed_difference = c[r][s] - c[s][r]
                    deviations[(r, s)].Start = abs(
                        fitted_start - observed_difference
                    )

        model.optimize()

        if model.SolCount == 0:
            return None, None

        x_values = {
            key: 1 if float(variable.X) >= 0.5 else 0
            for key, variable in x.items()
        }
        objective = evaluate_solution(n, g, c, x_values, weights)
        return x_values, objective

    finally:
        model.dispose()


# ---------------------------------------------------------------------------
# Phase 2: fixed bucket orders, optimize weights
# ---------------------------------------------------------------------------

def phase_2(
    n: int,
    g: int,
    c: List[List[float]],
    x_values: XValues,
    weights_start: Sequence[float],
    verbose: int,
    time_limit: float,
    environment: gp.Env,
) -> Tuple[Optional[List[float]], Optional[float]]:
    """Solve the weight-update LP for fixed bucket orders."""
    items = tuple(range(1, n + 1))
    groups = tuple(range(1, g + 1))
    model = gp.Model("OSBOP_PHASE_2", env=environment)

    try:
        model.setParam("OutputFlag", 1 if verbose else 0)
        model.setParam("LogToConsole", 1 if verbose else 0)
        set_threads_from_slurm(model)
        if time_limit > 0:
            model.setParam("TimeLimit", float(time_limit))

        weights = {
            component: model.addVar(
                lb=0.0,
                ub=1.0,
                name=f"w_{component}",
            )
            for component in groups
        }
        deviations = {
            (r, s): model.addVar(lb=0.0, name=f"v_{r}_{s}")
            for r in items
            for s in items
            if r < s
        }

        model.setObjective(
            gp.quicksum(deviations[(r, s)] for r in items for s in items if r < s),
            GRB.MINIMIZE,
        )
        model.addConstr(
            gp.quicksum(weights[component] for component in groups) == 1
        )
        for component in range(1, g):
            model.addConstr(weights[component] >= weights[component + 1])

        for r in items:
            for s in items:
                if r >= s:
                    continue

                fitted_difference = gp.quicksum(
                    weights[component]
                    * (
                        x_values[(r, s, component)]
                        - x_values[(s, r, component)]
                    )
                    for component in groups
                )
                observed_difference = c[r][s] - c[s][r]
                model.addConstr(
                    deviations[(r, s)]
                    >= fitted_difference - observed_difference
                )
                model.addConstr(
                    deviations[(r, s)]
                    >= observed_difference - fitted_difference
                )

        for component in groups:
            weights[component].Start = float(weights_start[component])

        for r in items:
            for s in items:
                if r >= s:
                    continue
                fitted_start = sum(
                    weights_start[component]
                    * (
                        x_values[(r, s, component)]
                        - x_values[(s, r, component)]
                    )
                    for component in groups
                )
                observed_difference = c[r][s] - c[s][r]
                deviations[(r, s)].Start = abs(
                    fitted_start - observed_difference
                )

        model.optimize()

        if model.SolCount == 0:
            return None, None

        new_weights = [0.0] * (g + 1)
        for component in groups:
            new_weights[component] = float(weights[component].X)

        objective = evaluate_solution(n, g, c, x_values, new_weights)
        return new_weights, objective

    finally:
        model.dispose()


# ---------------------------------------------------------------------------
# Multi-start initialization
# ---------------------------------------------------------------------------

def make_weight_starts(g: int, starts: int, seed: int) -> List[List[float]]:
    """Generate random nonincreasing vectors on the probability simplex."""
    generator = random.Random(seed)
    weight_starts: List[List[float]] = []

    for _ in range(starts):
        values = sorted(
            (generator.random() + 1e-9 for _ in range(g)),
            reverse=True,
        )
        total = sum(values)
        weights = [0.0] * (g + 1)
        for component in range(1, g + 1):
            weights[component] = values[component - 1] / total
        weight_starts.append(weights)

    return weight_starts


def random_linear_order_start(n: int, g: int, seed: int) -> XValues:
    """Generate feasible strict orders as warm starts for the weak-order MILP."""
    generator = random.Random(seed)
    items = list(range(1, n + 1))
    x_values: XValues = {}

    for component in range(1, g + 1):
        permutation = items.copy()
        generator.shuffle(permutation)
        position = {
            item: index for index, item in enumerate(permutation)
        }

        for r in items:
            for s in items:
                if r != s:
                    x_values[(r, s, component)] = (
                        1 if position[r] < position[s] else 0
                    )

    return x_values


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------

def append_summary(
    summaryfile: Path,
    instance_path: Path,
    instance: InstanceData,
    g: int,
    iterations: int,
    last_objective: Optional[float],
    best_objective: Optional[float],
    runtime: float,
    best_run: int,
    starts_used: int,
) -> None:
    """Append one matheuristic-result row to the shared summary file."""
    header = (
        f"{'instance':<30}{'n':>6}{'g':>6}{'D':>6}"
        f"{'it':>8}{'obj_last':>18}{'obj_best':>18}"
        f"{'runtime':>12}{'run':>6}{'starts':>8}\n"
    )

    last_value = float("nan") if last_objective is None else last_objective
    best_value = float("nan") if best_objective is None else best_objective
    line = (
        f"{instance_path.name:<30}{instance.n:>6d}{g:>6d}"
        f"{optional_int(instance.dispersion):>6}"
        f"{iterations:>8d}{last_value:>18.3f}{best_value:>18.3f}"
        f"{runtime:>12.3f}{best_run:>6d}{starts_used:>8d}\n"
    )
    append_text_locked(summaryfile, line, header=header)


def write_solution(
    outfile: Path,
    datafile: Path,
    instance: InstanceData,
    g: int,
    x_values: XValues,
    weights: Sequence[float],
    objective: float,
) -> None:
    """Append the best multi-start solution to the shared solution file."""
    lines = [f"{datafile}\n", f"n: {instance.n}\n"]
    lines.append(f"g: {g}\n\n")

    for component in range(1, g + 1):
        buckets, _ = extract_bucket_order(instance.n, x_values, component)
        validate_bucket_order(instance.n, x_values, component, buckets)

        lines.append(f"COMPONENT {component}\n")
        lines.append("Bucket order\n")
        lines.append(format_bucket_order(buckets) + "\n")
        lines.append("Weight\n")
        lines.append(f"{weights[component]:.10g}\n\n")

    lines.append(f"Total objective value: {fmt_real(objective, 5)}\n\n")
    append_text_locked(outfile, "".join(lines))


# ---------------------------------------------------------------------------
# Multi-start alternating-optimization procedure
# ---------------------------------------------------------------------------

def solve_osbop_heuristic_multistart(
    datafile: Path,
    outfile: Path,
    summaryfile: Path,
    g: int,
    starts: int,
    max_iterations: int,
    total_time_limit: float,
    phase_1_time_limit: float,
    tolerance: float,
    verbose_phases: int,
    seed: int,
    warmstart_dir: Path,
    require_warmstart: bool,
) -> None:
    """Run the complete multi-start alternating-optimization matheuristic."""
    if g < 1:
        raise ValueError("The number of bucket-order components must be positive.")
    if starts < 1:
        raise ValueError("The number of starts must be positive.")
    if max_iterations < 1:
        raise ValueError("The number of iterations must be positive.")

    instance = load_instance(datafile)
    c = build_normalized_matrix(instance)

    inherited_state: Optional[SolutionState] = None
    if g > 1:
        previous_path = state_path(warmstart_dir, datafile, g - 1)
        if previous_path.exists():
            inherited_state = load_inherited_state(
                previous_path, instance.n, g, c
            )
            print(
                f"Loaded inherited initialization from {previous_path.resolve()} "
                f"with objective {inherited_state.objective:.10g}."
            )
        elif require_warmstart:
            raise FileNotFoundError(
                f"Required warm start not found: {previous_path.resolve()}. "
                "Run the same instance with g-1 first or use "
                "--allow-missing-warmstart."
            )
        else:
            print(f"Warm start not found; using only random starts: {previous_path.resolve()}")

    environment = make_wls_env(max_retries=3)

    try:
        random_start_count = starts - (1 if inherited_state is not None else 0)
        random_weight_starts = make_weight_starts(g, random_start_count, seed)
        initializations: List[Tuple[List[float], Optional[XValues], bool]] = []
        if inherited_state is not None:
            initializations.append(
                (
                    list(inherited_state.weights),
                    dict(inherited_state.x_values),
                    True,
                )
            )
        initializations.extend(
            (weights, None, False) for weights in random_weight_starts
        )

        global_start_time = time.monotonic()
        starts_used = 0
        total_iterations = 0

        global_best_objective: Optional[float] = (
            inherited_state.objective if inherited_state is not None else None
        )
        global_best_x: Optional[XValues] = (
            dict(inherited_state.x_values) if inherited_state is not None else None
        )
        global_best_weights: Optional[List[float]] = (
            list(inherited_state.weights) if inherited_state is not None else None
        )
        global_best_run = 0
        global_best_last_objective: Optional[float] = global_best_objective

        for run_index, (initial_weights, initial_x, inherited_run) in enumerate(
            initializations, start=1
        ):
            elapsed = time.monotonic() - global_start_time
            remaining = total_time_limit - elapsed
            if remaining <= 0:
                break

            starts_used += 1
            if initial_x is None:
                current_x = random_linear_order_start(
                    instance.n,
                    g,
                    seed + 1000 * run_index,
                )
            else:
                current_x = dict(initial_x)
            current_weights = list(initial_weights)
            current_objective = evaluate_solution(
                instance.n,
                g,
                c,
                current_x,
                current_weights,
            )

            local_best_objective = current_objective
            local_best_x = dict(current_x)
            local_best_weights = list(current_weights)
            iterations_this_run = 0

            while iterations_this_run < max_iterations:
                elapsed = time.monotonic() - global_start_time
                remaining = total_time_limit - elapsed
                if remaining <= 0:
                    break

                objective_at_start = current_objective
                ranking_time_limit = min(phase_1_time_limit, remaining)

                new_x, phase_1_objective = phase_1(
                    n=instance.n,
                    g=g,
                    c=c,
                    weights=current_weights,
                    x_start=current_x,
                    verbose=verbose_phases,
                    time_limit=ranking_time_limit,
                    environment=environment,
                )
                if new_x is None or phase_1_objective is None:
                    break

                # The incumbent supplied as a MIP start is feasible for this
                # fixed-weight subproblem. Reject any deterioration caused by
                # an incomplete or numerically imprecise phase.
                if phase_1_objective > current_objective + 1e-7:
                    break

                current_x = new_x
                current_objective = phase_1_objective
                if current_objective < local_best_objective - 1e-12:
                    local_best_objective = current_objective
                    local_best_x = dict(current_x)
                    local_best_weights = list(current_weights)

                elapsed = time.monotonic() - global_start_time
                remaining = total_time_limit - elapsed
                if remaining <= 0:
                    break

                new_weights, phase_2_objective = phase_2(
                    n=instance.n,
                    g=g,
                    c=c,
                    x_values=current_x,
                    weights_start=current_weights,
                    verbose=verbose_phases,
                    time_limit=remaining,
                    environment=environment,
                )
                if new_weights is None or phase_2_objective is None:
                    break

                if phase_2_objective > current_objective + 1e-7:
                    break

                current_weights = new_weights
                current_objective = phase_2_objective
                iterations_this_run += 1
                total_iterations += 1

                if current_objective < local_best_objective - 1e-12:
                    local_best_objective = current_objective
                    local_best_x = dict(current_x)
                    local_best_weights = list(current_weights)

                improvement = objective_at_start - current_objective
                if improvement < tolerance:
                    break

            if (
                global_best_objective is None
                or local_best_objective < global_best_objective - 1e-12
                or (
                    inherited_run
                    and global_best_run == 0
                    and abs(local_best_objective - global_best_objective) <= 1e-12
                )
            ):
                global_best_objective = local_best_objective
                global_best_x = local_best_x
                global_best_weights = local_best_weights
                global_best_run = run_index
                global_best_last_objective = current_objective

        total_runtime = time.monotonic() - global_start_time

        append_summary(
            summaryfile=summaryfile,
            instance_path=datafile,
            instance=instance,
            g=g,
            iterations=total_iterations,
            last_objective=global_best_last_objective,
            best_objective=global_best_objective,
            runtime=total_runtime,
            best_run=global_best_run,
            starts_used=starts_used,
        )

        if (
            global_best_objective is None
            or global_best_x is None
            or global_best_weights is None
        ):
            print("No solution")
            return

        print(f"Writing BEST solution to: {outfile.resolve()}")
        write_solution(
            outfile=outfile,
            datafile=datafile,
            instance=instance,
            g=g,
            x_values=global_best_x,
            weights=global_best_weights,
            objective=global_best_objective,
        )

        incumbent_state = SolutionState(
            n=instance.n,
            g=g,
            objective=global_best_objective,
            weights=list(global_best_weights),
            x_values=dict(global_best_x),
        )
        output_state_path = state_path(warmstart_dir, datafile, g)
        save_solution_state(output_state_path, incumbent_state)
        print(f"Saved warm start to: {output_state_path.resolve()}")

    finally:
        environment.dispose()


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the multi-start OSBOP matheuristic."
    )
    parser.add_argument(
        "datafile",
        nargs="?",
        default=str(DEFAULT_DATAFILE),
        help="Path to the instance .dat file.",
    )
    parser.add_argument(
        "--g",
        type=int,
        default=DEFAULT_NUM_COMPONENTS,
        help="Number of bucket-order components.",
    )
    parser.add_argument(
        "--timelimit",
        type=float,
        default=DEFAULT_TOTAL_TIME_LIMIT,
        help="Global multi-start time limit in seconds.",
    )
    parser.add_argument(
        "--phase1-timelimit",
        type=float,
        default=DEFAULT_PHASE_1_TIME_LIMIT,
        help="Time limit for every ranking-update MILP.",
    )
    parser.add_argument(
        "--it_tot",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="Maximum alternating iterations per start.",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=DEFAULT_STARTS,
        help="Number of independent random starts.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="Convergence tolerance.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used by the multi-start initialization.",
    )
    parser.add_argument(
        "--outfile",
        default=str(DEFAULT_OUTFILE),
        help="Shared detailed solution file.",
    )
    parser.add_argument(
        "--summaryfile",
        default=str(DEFAULT_SUMMARYFILE),
        help="Shared computational summary file.",
    )
    parser.add_argument(
        "--warmstart-dir",
        default=str(DEFAULT_WARMSTART_DIR),
        help="Directory containing one machine-readable incumbent per instance and g.",
    )
    parser.add_argument(
        "--allow-missing-warmstart",
        action="store_true",
        help=(
            "Allow g>1 to run without the solution for g-1. This disables the "
            "cross-g monotonicity guarantee."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress Gurobi output from the two phases.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    arguments = parse_arguments()
    datafile = Path(arguments.datafile)

    print(f"SCRIPT: {Path(__file__).resolve()}")
    print(f"CWD: {Path.cwd()}")
    print(f"Reading instance: {datafile.resolve()}")
    print(f"SLURM_JOB_ID = {os.environ.get('SLURM_JOB_ID')}")
    print(f"SLURM_JOB_NODELIST = {os.environ.get('SLURM_JOB_NODELIST')}")
    print(f"SLURM_CPUS_PER_TASK = {os.environ.get('SLURM_CPUS_PER_TASK')}")

    if not datafile.exists():
        raise FileNotFoundError(f"Cannot find the .dat file: {datafile.resolve()}")

    solve_osbop_heuristic_multistart(
        datafile=datafile,
        outfile=Path(arguments.outfile),
        summaryfile=Path(arguments.summaryfile),
        g=arguments.g,
        starts=arguments.restarts,
        max_iterations=arguments.it_tot,
        total_time_limit=arguments.timelimit,
        phase_1_time_limit=arguments.phase1_timelimit,
        tolerance=arguments.tolerance,
        verbose_phases=0 if arguments.quiet else 1,
        seed=arguments.seed,
        warmstart_dir=Path(arguments.warmstart_dir),
        require_warmstart=not arguments.allow_missing_warmstart,
    )


if __name__ == "__main__":
    main()
