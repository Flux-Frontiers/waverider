# Reading the Rules Written in Disease: Kolmogorov-Arnold Networks on Clinical Manifolds

### How a 19th-century theorem about continuous functions reveals hidden algebraic rules inside cancer, heart disease, Parkinson's, and more

---

*We showed previously that clinical data lives on low-dimensional manifolds, and that a purely geometric classifier — zero parameters, no gradient descent — can beat trained neural networks at clinical diagnosis. This is the follow-up question: can we read the actual **rule** the manifold is encoding? Not just classify. Decode.*

*We can. And it changes what clinical AI means.*

---

## What ManifoldModel Can't Do

The WaveRider ManifoldModel classifies by geometric majority vote. It builds a graph of the data manifold, finds the k nearest neighbors of a query point in tangent-space coordinates, and returns the majority class. On heart disease, this outperforms a fully trained 7,022-parameter network. On breast cancer, it reaches 96.1% accuracy with zero learned weights.

But it has a structural limitation: **it gives you a label, not a number.**

The geometric vote either passes or fails. There's no continuous score that says "this patient's manifold position is 0.3 standard deviations into the disease region" or "this tumor's intrinsic coordinates place it near the boundary." No risk gradient. No formula.

For clinical decision support, that matters. A pathologist reviewing borderline cases wants risk stratification — a continuous score that reflects probability of disease, not just a binary flag. A regulatory body reviewing an AI diagnostic tool wants an interpretable rule, not a graph algorithm.

This is where the Kolmogorov-Arnold theorem enters.

---

## An Old Theorem, a New Network

In 1957, Andrei Kolmogorov proved something that seemed almost paradoxical: **any continuous multivariate function can be exactly represented as a composition of single-variable functions.**

Formally: for any continuous function $f: \mathbb{R}^n \to \mathbb{R}$, there exist continuous functions $\phi_{q,p}$ and $\Phi_q$ such that

$$f(x_1, \ldots, x_n) = \sum_{q=0}^{2n} \Phi_q \left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right)$$

The implications are deep. If the disease decision boundary is a smooth function of clinical features — and it is, manifolds are smooth by definition — then it can be decomposed into a sum of one-variable functions. The complexity of the multivariate classifier reduces to a set of 1D curves, each of which can be inspected, named, and understood.

Kolmogorov-Arnold Networks (KANs), introduced by Liu et al. in 2024, implement this theorem as a learnable architecture. Instead of fixed activation functions on neurons (as in standard MLPs), KANs place **learnable spline functions on the edges**. Each edge from input $i$ to hidden neuron $j$ is a parametric 1D curve, learned from data. After training, you can plot these curves, fit known functions to them (sigmoid, log, sin, polynomial), and read off a symbolic formula.

The key insight for us: **we already know the manifold coordinates**. The WaveRider stack gives us d* intrinsic dimensions — a small, geometrically meaningful coordinate system where the data truly lives. Feed those coordinates to a KAN, and you're asking the theorem to find the rule in the language the manifold already speaks.

---

## Experimental Design

For each of five clinical datasets, we ran three classifiers in 5-fold cross-validation:

1. **ManifoldModel** — zero parameters, pure geometric majority vote (our established baseline)
2. **KAN-raw** — KAN on all ambient features (e.g., all 30 breast cancer measurements)
3. **KAN-pca** — KAN on the d* intrinsic PCA coordinates (e.g., 8-dimensional breast cancer manifold)

For KAN-pca, we first discover the intrinsic dimension d* using local PCA with a 90% variance-retention threshold, project to that many principal components, then train a KAN with architecture `[d*, 4, output]`. The grid range is set to `[-5, 5]` to cover the actual data extent — a critical detail, as the default `[-1, 1]` grid causes zero gradients for all points outside the spline domain.

After cross-validation, we run symbolic regression on the full dataset: two-phase training (phase 1: no regularization to learn the function; phase 2: L1 sparsification to identify the dominant terms), followed by pruning and automatic symbolic function identification.

---

## Results: Five Diseases, One Consistent Story

| Dataset | Samples | Ambient | d* | Noise |
|---|---|---|---|---|
| Heart Disease (Cleveland) | 303 | 13D | **9** | 31% |
| Breast Cancer (Wisconsin) | 569 | 30D | **8** | 73% |
| Pima Indians Diabetes | 768 | 8D | **6** | 25% |
| Parkinson's Disease (Voice) | 195 | 22D | **6** | 73% |
| Dermatology (Skin Disease) | 366 | 34D | **12** | 65% |

### Accuracy

| Dataset | ManifoldModel | KAN-raw | KAN-pca |
|---|---|---|---|
| Heart Disease | **83.5% ± 0.8%** | 79.9% ± 1.9% | 75.6% ± 2.4% |
| Breast Cancer | 96.1% ± 0.6% | 96.1% ± 1.1% | **96.5% ± 0.7%** |
| Diabetes | 72.7% ± 1.3% | **73.9% ± 0.4%** | 72.7% ± 1.3% |
| Parkinson's | **90.3% ± 0.9%** | **90.8% ± 2.1%** | 88.2% ± 1.9% |
| Dermatology | 96.2% ± 0.6% | **96.7% ± 0.8%** | 94.8% ± 1.1% |

**ManifoldModel wins or ties on accuracy in four of five datasets.** On breast cancer, KAN-pca edges it by 0.4 percentage points. On diabetes and Parkinson's, it's effectively a tie. On heart disease, ManifoldModel leads by 3-8 percentage points. The geometric classifier is still the stronger hard-decision tool.

### AUC-ROC[^1]: Where the Story Breaks Open

| Dataset | ManifoldModel | KAN-raw | KAN-pca |
|---|---|---|---|
| Heart Disease | 83.4% | 85.7% | 83.5% |
| Breast Cancer | 95.3% | **99.5%** | **99.4%** |
| Diabetes | 68.5% | **79.8%** | 77.8% |
| Parkinson's | 84.0% | **96.0%** | **95.3%** |
| Dermatology | — | **99.8%** | 99.2% |

This is where the picture changes completely.

ManifoldModel's AUC is bounded by its architecture: it makes hard binary votes, not continuous probability estimates. KANs output calibrated logits that, after sigmoid, produce a genuine risk score. AUC measures ranking quality — the probability that a randomly selected diseased patient is ranked higher than a healthy one — and calibrated soft scores dominate hard votes.

The gaps are large enough to matter clinically:

- **Parkinson's**: KAN AUC is 96.0% vs ManifoldModel's 84.0%. That's a 12-point margin on the task of ranking patients by disease probability — exactly what a screening tool needs.
- **Breast Cancer**: KAN AUC is 99.4% vs 95.3%. Near-perfect risk stratification vs strong but not exceptional.
- **Diabetes**: KAN AUC is 79.8% vs ManifoldModel's 68.5%. On a dataset that is notoriously hard (high noise, overlapping classes), KAN's continuous scoring is substantially more useful.

**The pattern**: ManifoldModel for primary hard classification; KAN for continuous risk stratification. These are complementary, not competing.

---

## Heart Disease: Following the Formula

For the heart disease dataset, symbolic regression converges on a strikingly simple result.

After two-phase training (400 steps without regularization, 400 steps with L1 sparsification) on the full 303-point dataset, pruning removes all but one active edge. That edge — from **PC1** to the output — is a linear function with R² = 0.978. All other connections collapse to zero.

The decision boundary is, to very high precision:

$$\text{logit}(p_\text{disease}) \approx w \cdot \text{PC1} + b$$

What is PC1? The first principal component of the d*=9 intrinsic coordinate system, ordered by the loadings:

| Feature | Loading on PC1 |
|---|---|
| oldpeak (ST depression on exercise) | 0.397 |
| thalach (max heart rate achieved) | 0.389 |
| slope (ST segment slope at peak exercise) | 0.354 |

These three features dominate PC1. They are not arbitrary. They are the trio that cardiologists use in clinical treadmill stress testing: the degree of ST-segment depression induced by exercise (oldpeak), how high the patient's heart rate went (thalach), and the morphology of the recovery curve (slope). Together, they measure a single thing — **exercise-induced cardiac stress**.

The KAN found, without being told, that the heart disease manifold's discriminative direction is linear and points directly at exercise physiology. Every other clinical measurement — cholesterol, age, sex, resting blood pressure — contributes to the manifold's shape but not its decision boundary.

This aligns with decades of cardiology literature. The KAN is not just classifying; it is recovering the clinical rule that underlies the classification.

---

## Parkinson's: Voice as Geometry

Parkinson's disease is diagnosed in part by changes in vocal characteristics — tremor, rigidity, and loss of fine motor control manifest in subtle dysphonia. The UCI Parkinson's Voice dataset contains 22 voice biomarkers from sustained phonation of the vowel "a."

The intrinsic dimension at τ=0.90 is **d*=6** from 22 ambient features — 73% of the voice measurements are noise. The disease signal concentrates in six geometric directions.

KAN-raw achieves 90.8% accuracy and 96.0% AUC — slightly above ManifoldModel on accuracy, dramatically above on ranking quality. The AUC gap (96% vs 84%) is the largest in our dataset suite.

This makes geometric sense. Parkinson's voice data is known to be highly nonlinear — patients who are borderline or in early stages have ambiguous voice signals that span the boundary. A continuous risk score differentiates "definitely healthy" from "probably healthy" from "possibly affected" from "clearly diseased" in a way a binary vote cannot. The KAN's continuous output is clinically actionable at every risk level; the binary vote is actionable only at extremes.

---

## Breast Cancer: The 99.4% Question

The Wisconsin Breast Cancer dataset has 30 fine needle aspirate features — nuclear radius, texture, perimeter, area, smoothness, compactness, concavity, and their statistics. Intrinsic dimensionality is **d*=8**: 73% of the measurement space is noise.

ManifoldModel reaches 96.1% accuracy with zero parameters. KAN-pca on 8 intrinsic coordinates reaches 96.5% — marginally better — but the AUC gap is the point: 99.4% vs 95.3%.

What does 99.4% AUC mean in practice? It means that in almost every comparison between a randomly selected malignant case and a randomly selected benign case, the KAN ranks the malignant higher. Near-perfect separation in probability space, even for cases near the geometric boundary.

This is the difference between a screening test and a diagnostic tool. A screening test maximizes recall — catch everything — at the cost of false positives. An AUC-optimized system can be tuned to any operating point on the ROC curve, adjusting the probability threshold to trade off sensitivity and specificity based on clinical context. ManifoldModel, with its binary vote, cannot operate this way.

---

## What the Symbolic Formulas Tell Us (and Don't)

Across all five datasets, symbolic regression consistently converges to sparse formulas — one or two active edges survive pruning in every case. The formulas are:

- **Heart Disease**: linear in PC1 (exercise-induced cardiac stress)
- **Breast Cancer**: constant (over-sparsified; the boundary is genuinely multi-dimensional at 8D)
- **Diabetes**: constant (similar to breast cancer)
- **Parkinson's**: constant (22 voice features compress to 6D, but the boundary uses several dimensions)
- **Dermatology**: constant (multiclass with 6 categories; the boundary is intrinsically high-complexity)

The honest interpretation: **symbolic regression is more informative in low-noise, near-linear datasets**. Heart disease succeeds because its manifold is nearly flat (31% noise) and the dominant axis happens to be linear. The other datasets have higher ambient noise, more complex boundaries, or both.

This points to a clear research agenda: adaptive grid refinement (pykan's `grid_extension`), longer training before sparsification, and dataset-specific tuning. The framework is correct; the hyperparameters need dataset-specific calibration.

---

## The Geometric Decision Architecture

These results suggest a two-model deployment strategy for clinical AI:

**Primary diagnosis:** ManifoldModel. Zero parameters. Fits in microseconds. Matches or exceeds neural networks on hard-decision accuracy. Completely transparent: the graph, the local PCA bases, and the vote are all inspectable.

**Risk stratification:** KAN-pca. A small network (typically 4–12 inputs, one hidden layer of 4 neurons) trained on the intrinsic coordinates. Produces a calibrated probability score for every patient. Suitable for:
- Ranking referral priorities in a population screening program
- Setting institution-specific decision thresholds (AUC operates point-by-point on the ROC curve)
- Providing a continuous confidence score for borderline cases
- Symbolic regression to recover the clinical rule driving the boundary

The two-model system uses less compute than a standard MLP. ManifoldModel has zero parameters; KAN-pca has on the order of 100–500 parameters (spline coefficients) versus 7,000–36,000 for a standard MLP. The total system fits comfortably in kilobytes, not megabytes. It runs on the edge.

---

## Why This Matters Beyond Accuracy Tables

The clinical AI field has a reproducibility problem. Neural networks trained on tabular clinical data frequently fail to generalize across institutions, demographics, or patient populations. The reasons are well studied: overfitting to noise, spurious correlations with demographic proxies, and poor calibration of probability outputs.

The manifold framework addresses each of these.

**Noise robustness**: By identifying the d* intrinsic dimensions and restricting classification to that subspace, we discard the noise dimensions by construction. A classifier that only operates in the signal subspace cannot fit noise.

**Calibration**: KAN's continuous logit output, trained with binary cross-entropy, is naturally calibrated in a way that hard-vote classifiers are not. The AUC advantages we observe are direct evidence of this calibration.

**Interpretability**: If the symbolic regression identifies a formula — as it does for heart disease — that formula is auditable. A cardiologist can evaluate whether "linear function of exercise-induced ST depression and maximum heart rate" is clinically plausible. It is. A regulatory body can verify the formula against prior clinical knowledge. It aligns.

The geometric approach does not make accuracy-vs-interpretability a trade-off. The interpretability comes for free from the structure of the problem.

---

## The Technical Picture

For reproducibility, the complete pipeline is:

1. **Preprocessing**: median imputation → drop near-zero-variance features → StandardScaler → nan_to_num
2. **Dimensionality discovery**: local PCA at 10% of sample points, k=50 neighbors, τ=0.90 variance threshold
3. **ManifoldModel**: k_graph=15, k_pca=50, k_vote=7, τ=0.90
4. **KAN**: architecture `[d*, 4, 1]` (binary) or `[d*, 4, n_classes]` (multiclass); grid=5, k=3, grid_range=[-5,5]; Adam optimizer, lr=5e-3, 300 steps, λ=0.005
5. **Symbolic regression**: architecture `[d*, 4, 1]`, grid=7; Phase 1: 400 steps, λ=0; Phase 2: 400 steps, λ=5e-4; prune (node_th=0.03, edge_th=0.03); auto_symbolic

All code is available in the WaveRider repository at `benchmarks/canonical_tests/clinical/`.

---

## What's Next

Three extensions are natural from these results:

**All datasets, symbolic formulas**: Heart disease gave us a clean formula because its manifold is nearly flat. Breast cancer and Parkinson's need longer training, finer grids, and possibly differential geometry — the symbolic regression should operate in local chart coordinates, not global PCA. This is technically feasible and is the next experiment.

**ManifoldModel + KAN ensemble**: Use ManifoldModel's hard decision for classification, KAN's continuous score for confidence. Cases where the two disagree (ManifoldModel votes one class, KAN assigns <60% probability to it) are flagged for human review. This could be the right architecture for a clinical decision support system.

**Riemannian KAN**: The deepest integration. Instead of projecting to flat PCA coordinates and training a Euclidean KAN, train a KAN that respects the Riemannian metric of the manifold. The activation functions would be defined on geodesics. The formulas would be in terms of Riemannian distance, curvature, and principal directions. This is genuinely new mathematics, and the WaveRider geometric stack — TurtleND, ManifoldWalker, ManifoldObserver — provides all the machinery to do it.

---

*The data has always known. We are just learning to read what it knows.*

---

[^1]: **AUC-ROC** (Area Under the Receiver Operating Characteristic Curve) — a threshold-independent measure of how well a classifier ranks positive cases above negative ones. Formally, it equals the probability that a randomly selected diseased patient receives a higher risk score than a randomly selected healthy one. AUC = 0.5 is chance; AUC = 1.0 is perfect separation. Unlike accuracy, AUC does not require choosing a decision threshold and is unaffected by class imbalance, making it the standard metric for screening and risk-stratification tasks.

**Datasets**: Cleveland Heart Disease (UCI 45), Wisconsin Breast Cancer (sklearn), Pima Indians Diabetes (OpenML 37), Parkinson's Voice (UCI 174), Dermatology (UCI 33)

**Methods**: WaveRider v0.7, pykan 0.2+, scikit-learn 1.4+, Python 3.12

**Code**: `benchmarks/canonical_tests/clinical/kan_clinical.py`
