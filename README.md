<!DOCTYPE html>

<html lang="en">

<body>

<h1>A note on exact and matheuristic approaches to the Optimal Set of Bucket Orders Problem — Supporting Material</h1>

[![DOI](https://zenodo.org/badge/1317350489.svg)](https://doi.org/10.5281/zenodo.21720842)

<p>This repository provides the supplementary material for the article:</p>

<p><strong><em>“A note on exact and matheuristic approaches to the Optimal Set of Bucket Orders Problem”</em></strong></p>

<p><strong>Authors</strong><br>
Juan A. Aledo – Universidad de Castilla-La Mancha (UCLM), Spain<br>
Concepción Domínguez – Universidad de Murcia (UM), Spain<br>
Juan de Dios Jaime-Alcántara – Universidad Miguel Hernández de Elche (UMH), Spain<br>
Mercedes Landete – Universidad Miguel Hernández de Elche (UMH), Spain
</p>

<hr>

<h2>📁 Repository structure</h2>

<pre><code>inputs/
├── 1/
└── 2/
models/
outputs/
├── 1/
└── 2/
info_synthetic/
README.md
</code></pre>

<ul>
  <li><code>inputs/1/</code>: previously studied PrefLib benchmark instances in <code>.dat</code> format.</li>
  <li><code>inputs/2/</code>: synthetic instances <code>R1, R2, ..., R18</code> in <code>.dat</code> format.</li>
  <li><code>models/</code>: Python scripts implementing the compact exact formulation and the matheuristic presented in the article, together with the script <code>instance_generator.py</code> used to generate the synthetic instances.</li>
  <li><code>outputs/1/</code>: output files and computational summaries obtained for the previously studied PrefLib benchmark instances.</li>
  <li><code>outputs/2/</code>: output files and computational summaries obtained for the synthetic instances.</li>
  <li><code>info_synthetic/</code>: text files containing additional information on the synthetic instances, including the generating weak orders, weights, and component-specific pairwise count matrices.</li>
</ul>

<hr>

<h2>📄 Format of the <code>.dat</code> files</h2>

<p>The folders <code>inputs/1/</code> and <code>inputs/2/</code> contain the previously studied PrefLib benchmark instances and the synthetic instances, respectively. All files are provided in plain text format.</p>

<h3>Previously studied PrefLib instances</h3>

<p>Each benchmark instance contains:</p>

<ul>
  <li><code>n</code>: number of items</li>
  <li><code>m</code>: number of input orders</li>
  <li><code>a</code>: an <code>n × n</code> pairwise count matrix</li>
</ul>

<p>The general structure is:</p>

<pre><code>n: &lt;number_of_items&gt;
m: &lt;number_of_orders&gt;
a:
[
  &lt;row_1&gt;
  &lt;row_2&gt;
  ...
  &lt;row_n&gt;
]
</code></pre>

<h3>Synthetic instances</h3>

<p>Each synthetic instance contains:</p>

<ul>
  <li><code>n</code>: number of items</li>
  <li><code>g</code>: number of components used to generate the instance</li>
  <li><code>D</code>: maximum bucket-order distance used as noise in the generation process</li>
  <li><code>a</code>: an <code>n × n</code> pairwise count matrix</li>
</ul>

<p>The general structure is:</p>

<pre><code>n: &lt;number_of_items&gt;
g: &lt;number_of_components&gt;
D: &lt;maximum_bucket_order_distance&gt;
a:
[
  &lt;row_1&gt;
  &lt;row_2&gt;
  ...
  &lt;row_n&gt;
]
</code></pre>

<p>In the preprocessing step, for <code>r ≠ s</code>, the matrix <code>c</code> is computed from <code>a</code> as <code>c_rs = a_rs / (a_rs + a_sr)</code>, while <code>c_rr = 1/2</code>. Since a tie contributes <code>1/2</code> in each direction, the synthetic pairwise count matrices may contain half-integer values.</p>

<hr>

<h2>📄 Format of the synthetic information files</h2>

<p>Each file in <code>info_synthetic/</code> is provided in plain text format and is organized by components.</p>

<p>The general structure is:</p>

<pre><code>COMPONENT 1
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
</code></pre>

<p>For each component, the file contains:</p>

<ul>
  <li><code>Central weak order</code>: weak order used as the generating center</li>
  <li><code>Weight</code>: weight assigned to the component</li>
  <li><code>Pairwise count matrix</code>: matrix associated with the weak orders generated from that component</li>
</ul>

<p>Items enclosed in braces belong to the same bucket and are therefore tied.</p>

<hr>

<h2>📝 Citation</h2>

<p>If you use this material, please cite:</p>

<pre><code>@article{aledo2026exact,
  title   = {A note on exact and matheuristic approaches to the Optimal Set of Bucket Orders Problem},
  author  = {Aledo, Juan A. and Dom{\'i}nguez, Concepci{\'o}n and Jaime-Alc{\'a}ntara, Juan de Dios and Landete, Mercedes},
  journal = {Preprint},
  year    = {2026}
}
</code></pre>

</body>
</html>
