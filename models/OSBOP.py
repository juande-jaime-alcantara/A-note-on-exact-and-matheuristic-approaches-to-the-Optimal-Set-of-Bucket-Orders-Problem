#!/usr/bin/env python3
"""
Optimal Set of Bucket Orders Problem (OSBOP) Solver.

This script uses Gurobi to solve the OSBOP by parsing a custom `.dat` instance,
building the compact Mixed-Integer Linear Programming (MILP) formulation, and
writing both a detailed solution report and a summary of the execution. When
available, the solution obtained for one fewer component is used as a warm
start.
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
DEFAULT_OUTFILE = BASE_DIR / "../outputs/2/solutionsOSBOP.txt"
DEFAULT_SUMMARYFILE = BASE_DIR / "../outputs/2/summaryOSBOP.txt"
DEFAULT_WARMSTART_DIR = BASE_DIR / "../outputs/2/warmstarts/OSBOP"

# Default optimization parameters.
DEFAULT_NUM_COMPONENTS = 2
DEFAULT_TIME_LIMIT = 7200
DEFAULT_MIP_GAP = 1e-5


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
# Formatting and summaries
# ---------------------------------------------------------------------------

def fmt_real(value: float, max_decimals: int = 10) -> str:
    """Format a real value without unnecessary trailing zeroes."""
    text = f"{value:.{max_decimals}f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def optional_int(value: Optional[int]) -> str:
    """Format optional metadata in a summary table."""
    return "-" if value is None else str(value)


def safe_model_attribute(model: gp.Model, name: str, default: float) -> float:
    """Read a Gurobi attribute that may be unavailable for some statuses."""
    try:
        return float(getattr(model, name))
    except (AttributeError, gp.GurobiError):
        return default


def append_text_locked(
    path: Path,
    text: str,
    header: Optional[str] = None,
) -> None:
    """Append one complete block while holding an exclusive file lock.

    The experiments intentionally allow several SLURM jobs to write to the
    same result files. The lock prevents rows or solution blocks produced by
    different processes from being interleaved.
    """
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


def append_summary(
    summaryfile: Path,
    instance_path: Path,
    instance: InstanceData,
    g: int,
    model: gp.Model,
    time_limit: Optional[float],
) -> None:
    """Append one computational-result row to the summary file."""
    summaryfile = summaryfile.resolve()
    summaryfile.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"{'instance':<30}{'n':>6}{'g':>6}{'D':>6}"
        f"{'obj':>12}{'bound':>12}{'gap%':>10}{'sols':>8}"
        f"{'nodes':>12}{'status':>10}{'runtime':>12}\n"
    )

    solution_count = int(safe_model_attribute(model, "SolCount", 0.0))
    if solution_count > 0:
        objective = safe_model_attribute(model, "ObjVal", float("nan"))
        gap = safe_model_attribute(model, "MIPGap", float("nan"))
    else:
        objective = float("nan")
        gap = float("nan")

    bound = safe_model_attribute(model, "ObjBound", float("nan"))
    node_count = safe_model_attribute(model, "NodeCount", 0.0)
    runtime = safe_model_attribute(model, "Runtime", 0.0)
    status = int(safe_model_attribute(model, "Status", -1.0))
    if time_limit is not None and time_limit > 0:
        runtime = min(runtime, float(time_limit))

    line = (
        f"{instance_path.name:<30}{instance.n:>6d}{g:>6d}"
        f"{optional_int(instance.dispersion):>6}"
        f"{objective:>12.3f}{bound:>12.3f}{(100 * gap):>10.3f}"
        f"{solution_count:>8d}{node_count:>12.0f}{status:>10d}"
        f"{runtime:>12.3f}\n"
    )

    append_text_locked(summaryfile, line, header=header)


# ---------------------------------------------------------------------------
# Weak-order extraction
# ---------------------------------------------------------------------------

XVariables = Dict[Tuple[int, int, int], gp.Var]
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
    """Convert a collection of bucket orders into directed binary relations."""
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
            objective += abs(
                fitted_difference - (c[r][s] - c[s][r])
            )
    return objective


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


def solution_state_from_model(
    n: int,
    g: int,
    c: List[List[float]],
    x: XVariables,
    weights: Dict[int, gp.Var],
) -> SolutionState:
    """Extract a serializable solution from the current Gurobi incumbent."""
    x_values = {
        key: 1 if float(variable.X) >= 0.5 else 0
        for key, variable in x.items()
    }
    weight_values = [0.0] + [
        float(weights[component].X) for component in range(1, g + 1)
    ]
    objective = evaluate_solution(n, g, c, x_values, weight_values)
    return SolutionState(n, g, objective, weight_values, x_values)


def save_solution_state(path: Path, state: SolutionState) -> None:
    """Atomically save the incumbent for use by the run with g+1."""
    components: List[List[List[int]]] = []
    for component in range(1, state.g + 1):
        buckets, _ = extract_bucket_order_values(
            range(1, state.n + 1), state.x_values, component
        )
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


def apply_mip_start(
    n: int,
    g: int,
    c: List[List[float]],
    state: SolutionState,
    x: XVariables,
    weights: Dict[int, gp.Var],
    products: Dict[Tuple[int, int, int], gp.Var],
    deviations: Dict[Tuple[int, int], gp.Var],
) -> None:
    """Provide a complete feasible MIP start to the exact formulation."""
    for component in range(1, g + 1):
        weights[component].Start = state.weights[component]

    for key, variable in x.items():
        variable.Start = float(state.x_values[key])
        r, s, component = key
        products[(r, s, component)].Start = (
            state.weights[component] * state.x_values[key]
        )

    for r in range(1, n + 1):
        for s in range(r + 1, n + 1):
            fitted_difference = sum(
                state.weights[component]
                * (
                    state.x_values[(r, s, component)]
                    - state.x_values[(s, r, component)]
                )
                for component in range(1, g + 1)
            )
            deviations[(r, s)].Start = abs(
                fitted_difference - (c[r][s] - c[s][r])
            )


def extract_bucket_order(
    items: Sequence[int],
    x: XVariables,
    component: int,
) -> Tuple[List[List[int]], Dict[int, int]]:
    """Extract an ordered bucket partition from one integer solution.

    Twice the usual weak-order score is used. A strict win contributes 2, a
    tie contributes 1, and a strict loss contributes 0. Therefore, items in
    the same bucket have exactly the same integer score and no numerical
    rounding is required to identify ties.
    """
    score2: Dict[int, int] = {item: 0 for item in items}

    for r in items:
        for s in items:
            if r == s:
                continue

            x_rs = 1 if float(x[(r, s, component)].X) >= 0.5 else 0
            x_sr = 1 if float(x[(s, r, component)].X) >= 0.5 else 0

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


def extract_bucket_order_values(
    items: Sequence[int],
    x_values: XValues,
    component: int,
) -> Tuple[List[List[int]], Dict[int, int]]:
    """Extract ordered buckets from a dictionary of binary relations."""
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


def validate_extracted_order(
    items: Sequence[int],
    x: XVariables,
    component: int,
    buckets: Sequence[Sequence[int]],
) -> None:
    """Check that the extracted buckets reproduce every optimized relation."""
    bucket_position = {
        item: position
        for position, bucket in enumerate(buckets)
        for item in bucket
    }

    for r in items:
        for s in items:
            if r == s:
                continue

            expected = 1 if bucket_position[r] <= bucket_position[s] else 0
            observed = 1 if float(x[(r, s, component)].X) >= 0.5 else 0
            if expected != observed:
                raise RuntimeError(
                    "The extracted bucket order does not reproduce the "
                    f"optimized relation x[{r},{s},{component}]."
                )


def write_solution(
    outfile: Path,
    datafile: Path,
    instance: InstanceData,
    g: int,
    model: gp.Model,
    items: Sequence[int],
    groups: Sequence[int],
    x: XVariables,
    weights: Dict[int, gp.Var],
) -> None:
    """Append the best available OSBOP solution to a detailed output file."""
    lines = [f"{datafile}\n", f"n: {instance.n}\n"]
    lines.append(f"g: {g}\n\n")

    if model.SolCount == 0:
        lines.append("No solution found.\n\n")
        append_text_locked(outfile, "".join(lines))
        return

    lines.append(f"Objective value: {fmt_real(model.ObjVal, 5)}\n\n")

    for component in groups:
        buckets, _ = extract_bucket_order(items, x, component)
        validate_extracted_order(items, x, component, buckets)

        lines.append(f"COMPONENT {component}\n")
        lines.append("Bucket order\n")
        lines.append(format_bucket_order(buckets) + "\n")
        lines.append("Weight\n")
        lines.append(f"{weights[component].X:.10g}\n\n")

    lines.append("\n")
    append_text_locked(outfile, "".join(lines))


# ---------------------------------------------------------------------------
# Gurobi environment
# ---------------------------------------------------------------------------

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
# Exact OSBOP model
# ---------------------------------------------------------------------------

def solve_osbop(
    datafile: Path,
    outfile: Path,
    summaryfile: Path,
    g: int,
    time_limit: Optional[float],
    verbose: int,
    warmstart_dir: Path,
    require_warmstart: bool,
) -> None:
    """Build, solve, and report one exact OSBOP model."""
    if g < 1:
        raise ValueError("The number of bucket-order components must be positive.")

    instance = load_instance(datafile)
    c = build_normalized_matrix(instance)

    items = tuple(range(1, instance.n + 1))
    groups = tuple(range(1, g + 1))

    inherited_state: Optional[SolutionState] = None
    if g > 1:
        previous_path = state_path(warmstart_dir, datafile, g - 1)
        if previous_path.exists():
            inherited_state = load_inherited_state(
                previous_path, instance.n, g, c
            )
            print(
                f"Loaded warm start from {previous_path.resolve()} "
                f"with objective {inherited_state.objective:.10g}."
            )
        elif require_warmstart:
            raise FileNotFoundError(
                f"Required warm start not found: {previous_path.resolve()}. "
                "Run the same instance with g-1 first or use "
                "--allow-missing-warmstart."
            )
        else:
            print(f"Warm start not found; solving without it: {previous_path.resolve()}")

    environment = make_wls_env(max_retries=3)
    model: Optional[gp.Model] = None

    try:
        model = gp.Model("OSBOP", env=environment)
        model.setParam("MIPGap", DEFAULT_MIP_GAP)
        model.setParam("OutputFlag", 1 if verbose else 0)
        model.setParam("LogToConsole", 1 if verbose else 0)

        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus and slurm_cpus.isdigit():
            model.setParam("Threads", int(slurm_cpus))

        if time_limit is not None and time_limit > 0:
            model.setParam("TimeLimit", float(time_limit))

        x: XVariables = {
            (r, s, component): model.addVar(
                vtype=GRB.BINARY,
                name=f"x_{r}_{s}_{component}",
            )
            for r in items
            for s in items
            for component in groups
            if r != s
        }
        weights = {
            component: model.addVar(
                lb=0.0,
                ub=1.0,
                name=f"w_{component}",
            )
            for component in groups
        }
        products = {
            (r, s, component): model.addVar(
                lb=0.0,
                ub=1.0,
                name=f"u_{r}_{s}_{component}",
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

        # Completeness: either r weakly precedes s, s weakly precedes r,
        # or both relations hold when the items are tied.
        for component in groups:
            for r in items:
                for s in items:
                    if r < s:
                        model.addConstr(
                            x[(r, s, component)] + x[(s, r, component)] >= 1,
                            name=f"complete_{r}_{s}_{component}",
                        )

        # Transitivity of every weak order.
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
                            <= 1 + x[(r, t, component)],
                            name=f"trans_{r}_{s}_{t}_{component}",
                        )

        model.addConstr(
            gp.quicksum(weights[component] for component in groups) == 1,
            name="weight_sum",
        )

        # Remove symmetry among otherwise interchangeable components.
        for component in range(1, g):
            model.addConstr(
                weights[component] >= weights[component + 1],
                name=f"weight_order_{component}",
            )

        # Exact linearization products[r,s,i] = weights[i] * x[r,s,i].
        for component in groups:
            for r in items:
                for s in items:
                    if r == s:
                        continue
                    model.addConstr(
                        products[(r, s, component)] <= weights[component]
                    )
                    model.addConstr(
                        products[(r, s, component)] <= x[(r, s, component)]
                    )
                    model.addConstr(
                        products[(r, s, component)]
                        >= weights[component] - (1 - x[(r, s, component)])
                    )

        # Absolute deviations in the equivalent upper-triangular objective.
        for r in items:
            for s in items:
                if r >= s:
                    continue

                fitted_difference = gp.quicksum(
                    products[(r, s, component)]
                    - products[(s, r, component)]
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

        if inherited_state is not None:
            apply_mip_start(
                n=instance.n,
                g=g,
                c=c,
                state=inherited_state,
                x=x,
                weights=weights,
                products=products,
                deviations=deviations,
            )

        model.optimize()

        if inherited_state is not None:
            if model.SolCount == 0:
                raise RuntimeError(
                    "Gurobi did not retain the complete feasible warm start."
                )
            if model.ObjVal > inherited_state.objective + 1e-6:
                raise RuntimeError(
                    "The reported incumbent is worse than the inherited feasible "
                    f"solution: {model.ObjVal} > {inherited_state.objective}."
                )

        append_summary(
            summaryfile=summaryfile,
            instance_path=datafile,
            instance=instance,
            g=g,
            model=model,
            time_limit=time_limit,
        )

        print(f"STATUS = {model.Status} SOLCOUNT = {model.SolCount}")
        print(f"Writing solution to: {outfile.resolve()}")

        write_solution(
            outfile=outfile,
            datafile=datafile,
            instance=instance,
            g=g,
            model=model,
            items=items,
            groups=groups,
            x=x,
            weights=weights,
        )

        if model.SolCount > 0:
            incumbent_state = solution_state_from_model(
                n=instance.n,
                g=g,
                c=c,
                x=x,
                weights=weights,
            )
            output_state_path = state_path(warmstart_dir, datafile, g)
            save_solution_state(output_state_path, incumbent_state)
            print(f"Saved warm start to: {output_state_path.resolve()}")

    finally:
        try:
            if model is not None:
                model.dispose()
        finally:
            environment.dispose()


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Solve the exact OSBOP formulation with Gurobi."
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
        default=DEFAULT_TIME_LIMIT,
        help="Time limit in seconds; a nonpositive value disables the limit.",
    )
    parser.add_argument(
        "--outfile",
        default=str(DEFAULT_OUTFILE),
        help="Detailed solution output file.",
    )
    parser.add_argument(
        "--summaryfile",
        default=str(DEFAULT_SUMMARYFILE),
        help="Single-line computational summary file.",
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
        help="Suppress Gurobi console output.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    arguments = parse_arguments()
    datafile = Path(arguments.datafile)

    print(f"SCRIPT: {Path(__file__).resolve()}")
    print(f"CWD: {Path.cwd()}")
    print(f"Reading instance: {datafile.resolve()}")
    print(f"SLURM_CPUS_PER_TASK = {os.environ.get('SLURM_CPUS_PER_TASK')}")

    solve_osbop(
        datafile=datafile,
        outfile=Path(arguments.outfile),
        summaryfile=Path(arguments.summaryfile),
        g=arguments.g,
        time_limit=arguments.timelimit,
        verbose=0 if arguments.quiet else 1,
        warmstart_dir=Path(arguments.warmstart_dir),
        require_warmstart=not arguments.allow_missing_warmstart,
    )


if __name__ == "__main__":
    main()
