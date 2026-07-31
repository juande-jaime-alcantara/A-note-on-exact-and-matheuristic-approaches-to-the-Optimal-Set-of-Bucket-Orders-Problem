#!/usr/bin/env python3
"""
Optimal Set of Bucket Orders Problem (OSBOP) Synthetic Instance Generator.

This script generates the reproducible synthetic OSBOP instances used in the
computational experiments, together with text files describing their known
generating components.

A weak order is represented as an ordered list of nonempty buckets. Items in
the same bucket are tied, and every item in an earlier bucket precedes every
item in a later bucket.

The pairwise values 0, 1/2, and 1 are stored internally as 0, 1, and 2. This
avoids floating-point errors in distance calculations. Output files, however,
contain the actual bucket-matrix values and pairwise counts. Consequently,
dividing the matrix written under ``a:`` by ``NUMBER_OF_ORDERS`` gives the
normalized pair order matrix C. Equivalently, for r != s,

    c_rs = a_rs / (a_rs + a_sr).

The default parameters generate the 18 instances R1,...,R18 described in the
paper: n in {30,36,42}, generating component counts in {2,3,4}, and two weight
structures for every combination.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Experimental parameters
# ---------------------------------------------------------------------------

N_VALUES = (30, 36, 42)       # Numbers of items.
G_VALUES = (2, 3, 4)          # Numbers of generating components.
NUMBER_OF_ORDERS = 1000       # Weak orders sampled for each instance.
TIE_PROBABILITY = 0.25        # Probability of joining the preceding bucket.
DISPERSION_FRACTION = 0.05    # Radius as a fraction of the maximum distance.
SEPARATION_FRACTION = 0.20    # Initial minimum separation between centers.
SEPARATION_STEP_FRACTION = 0.01
MAX_CENTRAL_TRIES = 20000
MAX_MOVE_TRIES = 5000
SEED = 12345                  # Seed used for the reported instances.

# A weak order is an ordered list of buckets; every bucket is a list of items.
WeakOrder = list[list[int]]


# ---------------------------------------------------------------------------
# Basic weak-order operations
# ---------------------------------------------------------------------------

def copy_weak_order(order: WeakOrder) -> WeakOrder:
    """Return a deep copy of a weak order."""
    return [bucket.copy() for bucket in order]


def canonicalize(order: WeakOrder) -> WeakOrder:
    """Sort items inside each bucket, without changing the weak order."""
    return [sorted(bucket) for bucket in order]


def validate_weak_order(order: WeakOrder, n: int) -> None:
    """Raise ValueError if order is not a weak order of items 1,...,n."""
    if not order or any(not bucket for bucket in order):
        raise ValueError("Every weak order must contain nonempty buckets.")

    items = [item for bucket in order for item in bucket]
    if sorted(items) != list(range(1, n + 1)):
        raise ValueError("Every item from 1 to n must occur exactly once.")


def weak_order_to_matrix2(order: WeakOrder, n: int) -> np.ndarray:
    """Return twice the bucket matrix, with entries in {0,1,2}."""
    validate_weak_order(order, n)

    bucket_position = np.empty(n, dtype=np.int16)
    for position, bucket in enumerate(order):
        bucket_position[np.asarray(bucket, dtype=np.int16) - 1] = position

    row_position = bucket_position[:, None]
    column_position = bucket_position[None, :]

    # 2: row item precedes column item; 1: tie; 0: follows.
    matrix2 = np.where(
        row_position < column_position,
        2,
        np.where(row_position == column_position, 1, 0),
    )
    return matrix2.astype(np.int8)


def weak_order_distance2(
    matrix2_a: np.ndarray,
    matrix2_b: np.ndarray,
) -> int:
    """Return twice the pairwise distance between two bucket matrices.

    The corresponding undoubled distance is

        sum_{r<s} |b_rs - b'_rs|.

    Hence, changing a strict relation into a tie costs 1/2, and reversing a
    strict relation costs 1.
    """
    if matrix2_a.shape != matrix2_b.shape:
        raise ValueError("The two bucket matrices must have the same shape.")

    n = matrix2_a.shape[0]
    upper = np.triu_indices(n, k=1)
    difference = np.abs(
        matrix2_a.astype(np.int16) - matrix2_b.astype(np.int16)
    )
    return int(difference[upper].sum())


def format_half_integer(value2: int) -> str:
    """Format an integer stored at twice its actual value."""
    if value2 % 2 == 0:
        return str(value2 // 2)
    return f"{value2 / 2:.1f}"


def format_weak_order(order: WeakOrder) -> str:
    """Return a compact textual representation of a weak order."""
    return " > ".join(
        str(bucket[0])
        if len(bucket) == 1
        else "{" + " ".join(str(item) for item in bucket) + "}"
        for bucket in order
    )


# ---------------------------------------------------------------------------
# Central weak orders
# ---------------------------------------------------------------------------

def random_central_weak_order(
    n: int,
    tie_probability: float,
    rng: random.Random,
) -> WeakOrder:
    """Sample a central weak order from a random permutation.

    The permutation is uniform. Scanning it from left to right, every item
    after the first is merged into the preceding bucket with probability
    ``tie_probability``; otherwise, it starts a new bucket.
    """
    permutation = list(range(1, n + 1))
    rng.shuffle(permutation)

    order: WeakOrder = [[permutation[0]]]
    for item in permutation[1:]:
        if rng.random() < tie_probability:
            order[-1].append(item)
        else:
            order.append([item])

    return canonicalize(order)


def generate_central_weak_orders(
    n: int,
    g: int,
    tie_probability: float,
    rng: random.Random,
) -> tuple[list[WeakOrder], list[np.ndarray], int]:
    """Generate separated central weak orders by rejection sampling."""
    number_of_pairs = math.comb(n, 2)
    minimum_distance = math.ceil(SEPARATION_FRACTION * number_of_pairs)
    distance_step = math.ceil(SEPARATION_STEP_FRACTION * number_of_pairs)

    central_orders: list[WeakOrder] = []
    central_matrices2: list[np.ndarray] = []

    for _ in range(g):
        tries = 0
        while True:
            candidate = random_central_weak_order(
                n=n,
                tie_probability=tie_probability,
                rng=rng,
            )
            candidate_matrix2 = weak_order_to_matrix2(candidate, n)

            admissible = all(
                weak_order_distance2(candidate_matrix2, previous_matrix2)
                >= 2 * minimum_distance
                for previous_matrix2 in central_matrices2
            )

            if admissible:
                central_orders.append(candidate)
                central_matrices2.append(candidate_matrix2)
                break

            tries += 1
            if tries >= MAX_CENTRAL_TRIES:
                minimum_distance = max(0, minimum_distance - distance_step)
                tries = 0

    return central_orders, central_matrices2, minimum_distance


# ---------------------------------------------------------------------------
# Local perturbations
# ---------------------------------------------------------------------------

def propose_adjacent_item_move(
    order: WeakOrder,
    rng: random.Random,
) -> WeakOrder:
    """Generate a reversible local neighbor of a weak order.

    An item is selected uniformly. It is then either moved into an adjacent
    bucket or, if it is currently tied, extracted into a new singleton bucket
    immediately before or after its current bucket. The second possibility is
    the reverse of merging a singleton into an adjacent bucket and prevents the
    random walk from being biased toward orders with increasingly large ties.
    """
    n = sum(len(bucket) for bucket in order)
    selected_item = rng.randrange(1, n + 1)

    source_index = next(
        index
        for index, bucket in enumerate(order)
        if selected_item in bucket
    )

    moves: list[str] = []
    if source_index > 0:
        moves.append("join_left")
    if source_index + 1 < len(order):
        moves.append("join_right")
    if len(order[source_index]) > 1:
        moves.extend(("singleton_before", "singleton_after"))

    if not moves:
        raise RuntimeError("No local move is available for this weak order.")

    move = rng.choice(moves)
    neighbor = copy_weak_order(order)

    if move == "singleton_before":
        neighbor[source_index].remove(selected_item)
        neighbor.insert(source_index, [selected_item])

    elif move == "singleton_after":
        neighbor[source_index].remove(selected_item)
        neighbor.insert(source_index + 1, [selected_item])

    elif move == "join_left":
        neighbor[source_index].remove(selected_item)
        neighbor[source_index - 1].append(selected_item)
        if not neighbor[source_index]:
            del neighbor[source_index]

    elif move == "join_right":
        neighbor[source_index].remove(selected_item)
        if neighbor[source_index]:
            neighbor[source_index + 1].append(selected_item)
        else:
            del neighbor[source_index]
            neighbor[source_index].append(selected_item)

    return canonicalize(neighbor)


def sample_around_center(
    central_order: WeakOrder,
    central_matrix2: np.ndarray,
    n: int,
    dispersion: int,
    rng: random.Random,
) -> tuple[WeakOrder, np.ndarray, int]:
    """Sample a weak order inside the distance ball of radius dispersion.

    Starting from the center, exactly ``dispersion`` accepted local moves are
    performed. A proposal is accepted only if its distance from the center is
    at most ``dispersion``. As in the MLOP generator, accepted moves may partly
    undo previous moves, so the final distance need not equal the radius.
    """
    current_order = copy_weak_order(central_order)
    current_matrix2 = central_matrix2.copy()
    current_distance2 = 0

    for _ in range(dispersion):
        for _ in range(MAX_MOVE_TRIES):
            candidate_order = propose_adjacent_item_move(current_order, rng)
            candidate_matrix2 = weak_order_to_matrix2(candidate_order, n)
            candidate_distance2 = weak_order_distance2(
                candidate_matrix2,
                central_matrix2,
            )

            if candidate_distance2 <= 2 * dispersion:
                current_order = candidate_order
                current_matrix2 = candidate_matrix2
                current_distance2 = candidate_distance2
                break
        else:
            raise RuntimeError(
                "Unable to find an admissible perturbation after "
                f"{MAX_MOVE_TRIES} proposals."
            )

    return current_order, current_matrix2, current_distance2


# ---------------------------------------------------------------------------
# Weights and output files
# ---------------------------------------------------------------------------

def get_component_counts(g: int, weight_type: int) -> list[int]:
    """Return component sizes that sum to NUMBER_OF_ORDERS."""
    if g == 2:
        return [667, 333] if weight_type == 1 else [500, 500]
    if g == 3:
        return [571, 286, 143] if weight_type == 1 else [334, 333, 333]
    if g == 4:
        return [533, 267, 133, 67] if weight_type == 1 else [250] * 4
    raise ValueError("The number of generating components must be 2, 3, or 4.")


def write_half_integer_matrix(file, matrix2: np.ndarray) -> None:
    """Write a doubled internal matrix using its actual values."""
    for row in matrix2:
        file.write(
            "".join(
                format_half_integer(int(value2)).rjust(7)
                for value2 in row
            )
            + "\n"
        )


def write_dat_file(
    path: Path,
    n: int,
    g: int,
    dispersion: int,
    aggregated_counts2: np.ndarray,
) -> None:
    """Write an Rk.dat file.

    The field ``a`` stores the actual pairwise counts, possibly including
    half-integers because a tie contributes 1/2 in each direction. The usual
    normalization
    c_rs = a_rs / (a_rs + a_sr) therefore remains valid.
    """
    with path.open("w", encoding="ascii", newline="\n") as file:
        file.write(f"n: {n}\n")
        file.write(f"g: {g}\n")
        file.write(f"D: {dispersion}\n")
        file.write("a:\n")
        file.write("[ \n")
        write_half_integer_matrix(file, aggregated_counts2)
        file.write("]\n")


def write_summary_file(
    path: Path,
    g: int,
    component_counts: Sequence[int],
    central_orders: Sequence[WeakOrder],
    component_counts2: Sequence[np.ndarray],
) -> None:
    """Write a concise ground-truth summary for one instance."""
    with path.open("w", encoding="ascii", newline="\n") as file:
        for component in range(g):
            file.write(f"COMPONENT {component + 1}\n")
            file.write("Central weak order\n")
            file.write(format_weak_order(central_orders[component]) + "\n")
            file.write("Weight\n")
            file.write(
                f"{component_counts[component] / NUMBER_OF_ORDERS:.3f}\n"
            )
            file.write("Pairwise count matrix\n")
            write_half_integer_matrix(file, component_counts2[component])
            file.write("\n")


# ---------------------------------------------------------------------------
# Instance generation and validation
# ---------------------------------------------------------------------------

def validate_instance(
    n: int,
    g: int,
    dispersion: int,
    minimum_center_distance: int,
    component_counts: Sequence[int],
    central_matrices2: Sequence[np.ndarray],
    component_distances2: Sequence[Sequence[int]],
    aggregated_counts2: np.ndarray,
) -> None:
    """Check all structural and normalization properties of an instance."""
    if len(component_counts) != g or sum(component_counts) != NUMBER_OF_ORDERS:
        raise ValueError("Component counts do not define the required population.")

    for first in range(g):
        for second in range(first + 1, g):
            distance2 = weak_order_distance2(
                central_matrices2[first],
                central_matrices2[second],
            )
            if distance2 < 2 * minimum_center_distance:
                raise ValueError("Central weak orders violate their separation.")

    for distances2 in component_distances2:
        if len(distances2) == 0 or max(distances2) > 2 * dispersion:
            raise ValueError("A generated weak order violates the radius D.")

    if aggregated_counts2.shape != (n, n):
        raise ValueError("The aggregated matrix has an invalid shape.")

    if np.any(np.diag(aggregated_counts2) != NUMBER_OF_ORDERS):
        raise ValueError("The diagonal of the doubled count matrix is invalid.")

    expected_pair_sum = 2 * NUMBER_OF_ORDERS
    for r in range(n):
        for s in range(r + 1, n):
            if int(aggregated_counts2[r, s] + aggregated_counts2[s, r]) != expected_pair_sum:
                raise ValueError("The pairwise count matrix is not normalized.")


def generate_instance(
    n: int,
    g: int,
    instance_name: str,
    weight_type: int,
    output_directory: Path,
    rng: random.Random,
) -> None:
    """Generate one instance and write its data and summary files."""
    number_of_pairs = math.comb(n, 2)
    dispersion = math.ceil(DISPERSION_FRACTION * number_of_pairs)
    component_counts = get_component_counts(g, weight_type)

    central_orders, central_matrices2, minimum_center_distance = (
        generate_central_weak_orders(
            n=n,
            g=g,
            tie_probability=TIE_PROBABILITY,
            rng=rng,
        )
    )

    component_pairwise_counts2: list[np.ndarray] = []
    component_distances2: list[list[int]] = []

    for component in range(g):
        pairwise_counts2 = np.zeros((n, n), dtype=np.int32)
        distances2: list[int] = []

        for _ in range(component_counts[component]):
            _, sampled_matrix2, distance2 = sample_around_center(
                central_order=central_orders[component],
                central_matrix2=central_matrices2[component],
                n=n,
                dispersion=dispersion,
                rng=rng,
            )
            pairwise_counts2 += sampled_matrix2
            distances2.append(distance2)

        component_pairwise_counts2.append(pairwise_counts2)
        component_distances2.append(distances2)

    aggregated_counts2 = np.sum(
        np.stack(component_pairwise_counts2, axis=0),
        axis=0,
        dtype=np.int32,
    )

    validate_instance(
        n=n,
        g=g,
        dispersion=dispersion,
        minimum_center_distance=minimum_center_distance,
        component_counts=component_counts,
        central_matrices2=central_matrices2,
        component_distances2=component_distances2,
        aggregated_counts2=aggregated_counts2,
    )

    write_dat_file(
        path=output_directory / f"{instance_name}.dat",
        n=n,
        g=g,
        dispersion=dispersion,
        aggregated_counts2=aggregated_counts2,
    )
    write_summary_file(
        path=output_directory / f"Summary{instance_name}.txt",
        g=g,
        component_counts=component_counts,
        central_orders=central_orders,
        component_counts2=component_pairwise_counts2,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_output = Path(__file__).resolve().parent / "synthetic_osbop"
    parser = argparse.ArgumentParser(
        description="Generate the synthetic OSBOP instances R1,...,R18."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory in which the .dat and summary files are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Seed of the pseudorandom generator.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate all default instances."""
    arguments = parse_arguments()
    output_directory = arguments.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    rng = random.Random(arguments.seed)
    instance_counter = 1

    for n in N_VALUES:
        for g in G_VALUES:
            for weight_type in (1, 2):
                instance_name = f"R{instance_counter}"
                generate_instance(
                    n=n,
                    g=g,
                    instance_name=instance_name,
                    weight_type=weight_type,
                    output_directory=output_directory,
                    rng=rng,
                )
                print(
                    f"Generated {instance_name}: n={n}, g={g}, "
                    f"D={math.ceil(DISPERSION_FRACTION * math.comb(n, 2))}, "
                    f"weight_type={weight_type}"
                )
                instance_counter += 1


if __name__ == "__main__":
    main()
