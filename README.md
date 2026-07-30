# Exact and matheuristic approaches for the Optimal Set of Bucket Orders Problem - Supporting Material

This repository provides the supplementary material for the article:

**_“Exact and matheuristic approaches for the Optimal Set of Bucket Orders Problem”_**

**Authors**  
Juan A. Aledo - Universidad de Castilla-La Mancha (UCLM), Spain  
Concepción Domínguez - Universidad de Murcia (UM), Spain  
Juan de Dios Jaime-Alcántara - Universidad Miguel Hernández de Elche (UMH), Spain  
Mercedes Landete - Universidad Miguel Hernández de Elche (UMH), Spain

---

## 📁 Repository structure

```text
inputs/
├── benchmark/
└── synthetic/
models/
├── OSBOP_exact.py
├── OSBOP_matheuristic.py
└── generate_osbop_instances.py
outputs/
├── exact/
├── matheuristic/
└── solver_logs/
info_synthetic/
recovery/
README.md
```

- `inputs/benchmark/`: benchmark instances derived from the PrefLib datasets considered in the article.
- `inputs/synthetic/`: synthetic instances `R1, R2, ..., R18` used in the scalability and recovery experiments.
- `models/`: Python implementations of the exact MILP formulation and the multi-start alternating-optimization matheuristic, together with the synthetic instance generator.
- `outputs/exact/`: detailed solutions and computational summaries obtained with the exact formulation.
- `outputs/matheuristic/`: detailed solutions and computational summaries obtained with the matheuristic.
- `outputs/solver_logs/`: raw solver outputs from the computational experiments.
- `info_synthetic/`: generating weak orders, weights, and component-specific pairwise count matrices for the synthetic instances.
- `recovery/`: complete comparison between the generating and recovered components for the synthetic test set.

---

## 📄 Format of the `.dat` files

The instances are provided in plain text format. Each file contains an `n × n` pairwise count matrix `a`.

### Benchmark instances

The benchmark files contain:

- `n`: number of items;
- `m`, when present: number of input orders;
- `a`: pairwise count matrix.

Their general structure is:

```text
n: <number_of_items>
m: <number_of_orders>
a:
[
  <row_1>
  <row_2>
  ...
  <row_n>
]
```

### Synthetic instances

Each synthetic file additionally contains:

- `g`: number of components used to generate the instance;
- `D`: maximum bucket-order distance from a generated weak order to its generating central weak order.

Their general structure is:

```text
n: <number_of_items>
g: <number_of_generating_components>
D: <maximum_bucket_order_distance>
a:
[
  <row_1>
  <row_2>
  ...
  <row_n>
]
```

For $r \neq s$, the normalized pair-order matrix used by the optimization models is computed as

$$
c_{rs}=\frac{a_{rs}}{a_{rs}+a_{sr}},
$$

while $c_{rr}=1/2$. Because a tie contributes $1/2$ in each direction, the synthetic pairwise count matrices may contain half-integer values.

---

## 📄 Synthetic instance information

Each file in `info_synthetic/` records the known generating structure of one synthetic instance. Its general format is:

```text
COMPONENT 1
Central weak order
...
Weight
...
Pairwise count matrix
...

COMPONENT 2
Central weak order
...
Weight
...
Pairwise count matrix
...

...
```

For each component, the file contains:

- `Central weak order`: the weak order used as the component center;
- `Weight`: its generating mixture weight;
- `Pairwise count matrix`: the pairwise information contributed by the weak orders sampled from that component.

The buckets are listed from top to bottom. Items enclosed in braces belong to the same bucket and are therefore tied.

---

## ⚙️ Generating the synthetic instances

The script `models/generate_osbop_instances.py` implements the generation procedure used in the computational experiments. The default parameters generate the 18 instances `R1, R2, ..., R18`, with:

- $n \in \{30,36,42\}$;
- $g \in \{2,3,4\}$;
- 1000 generated weak orders per instance;
- equal or decreasing component weights;
- dispersion $D=\left\lceil 0.05\binom{n}{2}\right\rceil$.

To reproduce the instances from the root directory, run:

```bash
python models/generate_osbop_instances.py --output-dir generated_instances
```

The command creates the `.dat` files and their corresponding `SummaryR*.txt` files. The copies distributed with this repository are organized under `inputs/synthetic/` and `info_synthetic/`, respectively.

---

## ▶️ Using the optimization models

The implementations require Python, NumPy, Gurobi 12.0.1, its Python interface `gurobipy`, and a valid Gurobi license.

The available command-line options can be inspected with:

```bash
python models/OSBOP_exact.py --help
python models/OSBOP_matheuristic.py --help
```

For example, to solve synthetic instance `R3` with three components:

```bash
python models/OSBOP_exact.py inputs/synthetic/R3.dat --g 3
python models/OSBOP_matheuristic.py inputs/synthetic/R3.dat --g 3 --allow-missing-warmstart
```

The matheuristic can inherit the solution obtained for $g-1$ as one of the initializations for $g$. Therefore, configurations for the same instance may also be run sequentially in increasing order of $g$. Detailed solutions, computational summaries, and raw solver outputs from the experiments reported in the article are provided in `outputs/`.

---

## 🔎 Recovery results

The controlled synthetic instances permit an ex post comparison between the generating and recovered components. For every instance, recovery is assessed using the best solution obtained with the same number of components as the generating model.

Since component labels are interchangeable, the recovered components are matched with the generating components by minimizing their total bucket-order distance $d_{\mathrm{BO}}$. The files in `recovery/` report:

- the generating and recovered component weights;
- the absolute weight errors;
- the bucket-order distance $d_{\mathrm{BO}}$;
- Kendall's rank correlation coefficient adjusted for ties, $\tau_b$.

The article presents instance `R3` as a detailed illustration, while this repository provides the complete recovery results for the synthetic test set.

---

## 📝 Citation

If you use this material, please cite:

```bibtex
@article{aledo2026exact,
  title   = {Exact and matheuristic approaches for the Optimal Set of Bucket Orders Problem},
  author  = {Aledo, Juan A. and Dom{\'i}nguez, Concepci{\'o}n and Jaime-Alc{\'a}ntara, Juan de Dios and Landete, Mercedes},
  journal = {Preprint},
  year    = {2026}
}
```
