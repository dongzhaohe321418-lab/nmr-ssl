# Sort-Match Theorem: Bipartite Matching → Sorting under 1D Convex Cost

**Status:** v0.1 formal. Load-bearing for the paper. Every step double-checked.

---

## 1. Setting

A molecule has $n$ atoms (of a given nucleus, e.g., ¹³C). A model $f_\theta$ produces a vector of predicted chemical shifts
$$\hat{\mathbf{y}} = (\hat{y}_1, \ldots, \hat{y}_n) \in \mathbb{R}^n.$$

A literature-extracted NMR spectrum provides an **unassigned** ground-truth set of $n$ peaks
$$\mathbf{y}^\star = \{y_1^\star, \ldots, y_n^\star\} \subset \mathbb{R}$$
(a multiset of real-valued shifts; we assume for now that the number of peaks equals the number of atoms — degeneracy from chemical equivalence is handled in §6).

**Problem.** Because the peaks are unassigned, the supervision signal should be invariant to any re-ordering of the target vector. The natural permutation-invariant loss is the optimal bipartite matching:

$$\mathcal{L}_{\text{match}}(\hat{\mathbf{y}}, \mathbf{y}^\star) \;=\; \min_{\sigma \in \mathfrak{S}_n} \; \sum_{i=1}^{n} \phi\!\left(\hat{y}_i - y_{\sigma(i)}^\star\right)$$

where $\phi: \mathbb{R} \to \mathbb{R}_{\geq 0}$ is a per-atom loss (e.g., $\phi(t) = |t|$ for MAE, $\phi(t) = t^2$ for MSE).

Solving this naively via the Hungarian algorithm costs $O(n^3)$ per molecule. For training on millions of molecules with $n$ up to a few hundred, this is prohibitive; worse, it breaks GPU batching because Hungarian is combinatorial and non-differentiable.

**Claim.** Under a mild convexity assumption on $\phi$, the optimal matching is the one obtained by **independently sorting** $\hat{\mathbf{y}}$ and $\mathbf{y}^\star$ and aligning them in order. This reduces the matching step from $O(n^3)$ to $O(n \log n)$, is GPU-friendly (standard `torch.sort`), and is differentiable almost everywhere through straight-through gradients on the sort permutation.

---

## 2. Main theorem

> **Theorem 1 (Sort-Match Optimality).**
> Let $\hat{\mathbf{y}}, \mathbf{y}^\star \in \mathbb{R}^n$ and let $\phi: \mathbb{R} \to \mathbb{R}$ be a convex function. Let $\hat{\mathbf{y}}_{\uparrow}$ and $\mathbf{y}^\star_{\uparrow}$ denote the non-decreasing re-orderings of the two vectors. Then
> $$\min_{\sigma \in \mathfrak{S}_n} \sum_{i=1}^n \phi\!\left(\hat{y}_{\sigma(i)} - y_i^\star\right) \;=\; \sum_{i=1}^n \phi\!\left(\hat{y}_{\uparrow,i} - y^\star_{\uparrow,i}\right).$$
>
> Equivalently: the optimal bipartite matching between two finite sets of reals, under any convex cost, is obtained by sorting both sides in the same order.

This is a classical result with roots in Hardy–Littlewood–Pólya's *Inequalities* (1934, Theorem 368 and surrounding material on the "rearrangement inequality"). We provide a self-contained short proof for completeness, because we need to cite the theorem precisely and use $\phi$ in a slightly broader form than the standard statement.

---

## 3. Proof

### 3.1 Reduction to adjacent transpositions

Any permutation can be written as a product of adjacent transpositions (this is the elementary "bubble sort" argument). If we can show that any *single* adjacent transposition that moves the permutation closer to the sorted alignment does not increase the sum, then induction on the number of transpositions gives the result.

### 3.2 The two-pair swap lemma

> **Lemma 1 (Two-Pair Swap).**
> Let $a_1 \leq a_2$ and $b_1 \leq b_2$ be real numbers, and let $\phi: \mathbb{R} \to \mathbb{R}$ be convex. Then
> $$\phi(a_1 - b_1) + \phi(a_2 - b_2) \;\leq\; \phi(a_1 - b_2) + \phi(a_2 - b_1).$$
> In words: the "sorted" pairing has smaller (or equal) total cost than the "swapped" pairing.

**Proof.** Define $u = a_1 - b_1$, $v = a_2 - b_2$, $x = a_1 - b_2$, $y = a_2 - b_1$. We note three facts:

1. **Same sum.** $u + v = (a_1 - b_1) + (a_2 - b_2) = (a_1 + a_2) - (b_1 + b_2) = (a_1 - b_2) + (a_2 - b_1) = x + y.$
2. **Order of pairs.** From $a_1 \leq a_2$ and $b_1 \leq b_2$:
   - $x = a_1 - b_2 \leq a_1 - b_1 = u$ (subtracting a larger value from $a_1$),
   - $y = a_2 - b_1 \geq a_2 - b_2 = v$ (subtracting a smaller value from $a_2$).
   - So $x \leq u$ and $v \leq y$, and combined with $x + y = u + v$, we conclude that $(x, y)$ is *more spread out* than $(u, v)$: $\min(x, y) \leq \min(u, v)$ and $\max(x, y) \geq \max(u, v)$.
3. **Majorization.** Any two-element pair whose minimum is smaller and whose maximum is larger, while the sum is equal, **majorizes** the other pair in the sense of Hardy–Littlewood–Pólya. Formally: $(x, y) \succ (u, v)$.

By the Hardy–Littlewood–Pólya majorization inequality (a fundamental property of convex functions): for any convex $\phi$ and any vectors $(p_1, \ldots, p_n)$ majorizing $(q_1, \ldots, q_n)$,
$$\sum_{i=1}^n \phi(p_i) \geq \sum_{i=1}^n \phi(q_i).$$

Applying this with $n=2$, $(p_1, p_2) = (x, y)$, $(q_1, q_2) = (u, v)$:
$$\phi(x) + \phi(y) \geq \phi(u) + \phi(v),$$
which is exactly
$$\phi(a_1 - b_2) + \phi(a_2 - b_1) \geq \phi(a_1 - b_1) + \phi(a_2 - b_2). \qquad\square$$

A self-contained proof of the two-element majorization inequality without invoking HLP: write
$u = \lambda x + (1-\lambda) y$ for some $\lambda \in [0,1]$ (possible because $u$ lies in the closed interval $[\min(x,y), \max(x,y)] = [x,y]$, using fact 2 above — wait, that gives $x \leq u$, but we also need $u \leq y$. Is $u \leq y$? $u - y = (a_1 - b_1) - (a_2 - b_1) = a_1 - a_2 \leq 0$, yes. So $x \leq u \leq y$. Similarly $x \leq v \leq y$.). Then $v = (1-\lambda) x + \lambda y$ (by the sum constraint $u + v = x + y$). By convexity,
$$\phi(u) \leq \lambda \phi(x) + (1-\lambda)\phi(y), \qquad \phi(v) \leq (1-\lambda)\phi(x) + \lambda \phi(y).$$
Adding: $\phi(u) + \phi(v) \leq \phi(x) + \phi(y)$. Done — no need to even name "HLP" or "majorization."

### 3.3 Bubble-sort induction

Let $\sigma^\star$ be any permutation minimizing $\sum_i \phi(\hat{y}_{\sigma(i)} - y_i^\star)$. If $\sigma^\star$ is not the one that aligns the two sorted orderings (call it $\sigma_{\text{sort}}$), then there exists a pair of indices $i < j$ with $\hat{y}_{\sigma^\star(i)} > \hat{y}_{\sigma^\star(j)}$ while $y_i^\star \leq y_j^\star$ (some "inversion" relative to the sorted pairing). Swap the assignment at $i$ and $j$: by Lemma 1 (applied to $a_1 = \hat{y}_{\sigma^\star(j)}$, $a_2 = \hat{y}_{\sigma^\star(i)}$, $b_1 = y_i^\star$, $b_2 = y_j^\star$), the total loss weakly decreases. Each such swap strictly reduces the number of inversions relative to $\sigma_{\text{sort}}$, so after finitely many swaps we reach $\sigma_{\text{sort}}$ without ever increasing the sum. Hence $\sigma_{\text{sort}}$ achieves the minimum. $\quad\square$

---

## 4. Corollaries

> **Corollary 1 (Standard losses).** For $\phi(t) = |t|^p$ with $p \geq 1$ (convex for $p \geq 1$), Theorem 1 applies. In particular:
> - **MAE** ($p=1$): optimal bipartite matching is the sorted assignment.
> - **MSE** ($p=2$): optimal bipartite matching is the sorted assignment.
> - **Huber loss** (piecewise quadratic/linear, convex): optimal bipartite matching is the sorted assignment.

> **Corollary 2 (Computational reduction).** Under the hypotheses of Theorem 1, the permutation-invariant set-supervision loss
> $$\mathcal{L}_{\text{match}}(\hat{\mathbf{y}}, \mathbf{y}^\star) = \min_\sigma \sum_i \phi\!\left(\hat{y}_{\sigma(i)} - y_i^\star\right)$$
> can be computed in $O(n \log n)$ time via
> $$\mathcal{L}_{\text{sort}}(\hat{\mathbf{y}}, \mathbf{y}^\star) = \sum_i \phi\!\left(\hat{y}_{\uparrow,i} - y^\star_{\uparrow,i}\right),$$
> with **exact equality**, not an approximation.

> **Corollary 3 (Gradient structure).** The gradient of $\mathcal{L}_{\text{sort}}$ with respect to $\hat{\mathbf{y}}$ is, at any point where the sorting permutation is unique:
> $$\frac{\partial \mathcal{L}_{\text{sort}}}{\partial \hat{y}_i} = \phi'\!\left(\hat{y}_i - y^\star_{\pi(i)}\right),$$
> where $\pi$ is the permutation that sorts $\hat{y}$ into the order of the sorted $\mathbf{y}^\star$. At ties (zero-measure set), any sub-gradient selection works. Standard frameworks (`torch.sort`) handle this correctly via gather/scatter.

---

## 5. Relationship to DETR's Hungarian loss

DETR (Carion et al., ECCV 2020) uses a Hungarian matcher between predicted and target *object* sets, where each object is a $(class, box)$ pair and the per-pair cost is a multi-dimensional function involving classification scores, L1 box distance, and GIoU. That cost is **not** a function of a single scalar difference, so Theorem 1 does not apply — the Hungarian step is genuinely needed there, and DETR pays the $O(N^3)$ cost.

Our setting is structurally simpler: the targets are 1D scalars (ppm values) and the cost decomposes as a convex function of the scalar difference. This 1D structure is what makes the reduction possible. **This is the methodological observation that unlocks scalable literature-scale training.**

We should cite DETR as the canonical set-prediction framework and state clearly:

> *"Unlike DETR's multi-dimensional object prediction, chemical shift prediction is intrinsically 1D — each target is a single scalar in ppm. We exploit this 1D structure: Theorem 1 shows that the $O(N^3)$ Hungarian matcher collapses to an $O(N \log N)$ sort without any loss of optimality, provided the per-atom cost is convex. This makes set-based supervision practical at the 10⁵-10⁶ training-example scale we target."*

---

## 6. Limitations and extensions

### 6.1 Unequal cardinalities (chemical-equivalence degeneracy)

In real NMR, multiple atoms in the same chemical environment produce a single peak (e.g., the three H's of a methyl group appear as one peak). So the number of predicted atoms $n$ may exceed the number of observed peaks $m$.

**Handling in v1 (MVP).** We restrict to molecules where $n = m$ (no degenerate environments, or the degeneracy has been resolved in pre-processing). This loses perhaps 20–30% of NMRShiftDB2 molecules but is honest and simple. Discuss as a limitation.

**Handling in the future version.** The natural extension is *multi-set matching with multiplicities*: each target peak has an integer multiplicity $k_j$ and we need to assign $k_j$ predicted atoms to it. Theorem 1 generalizes: if we "explode" each target with multiplicity $k_j$ into $k_j$ copies of the same scalar, the resulting equal-cardinality problem still has a sorted optimal solution, by the same proof. The only subtlety is that ties among targets permit many optimal assignments, which are all equal in loss. This is fine for training.

**Harder case:** when multiplicities are not known from the spectrum itself (peak integration is ambiguous). This is a research problem in its own right and is left as future work.

### 6.2 Missing peaks and noise

Literature-extracted spectra have missing or noisy peaks. A robust version of the loss uses a *trimmed* or *capped* sort-match, e.g., dropping the worst $k$ matches before summing. This preserves Theorem 1 for the retained matches (the trimmed sort of two sequences is still a legitimate truncated sort). Detailed analysis deferred.

### 6.3 Convexity is load-bearing

If $\phi$ is **not convex** (e.g., truncated quadratic used for robustness, or log-cosh outside of the convex regime), Theorem 1 fails in general. The reduction is specifically a convexity consequence. This is why the paper must be careful to (a) use convex surrogates at training time and (b) only use non-convex losses at *evaluation* time.

### 6.4 Differentiability at ties

`torch.sort` is differentiable via the gather-scatter trick. At tied values the sorting permutation is not unique, but the loss value is well-defined (all permutations yield the same value by symmetry). Gradients are sub-gradients on a zero-measure set.

---

## 7. Sanity check: numerical verification

We will verify Theorem 1 numerically in `tests/test_theorem.py` by:
1. Generating random $\hat{\mathbf{y}}, \mathbf{y}^\star$ pairs,
2. Computing the true optimal matching via `scipy.optimize.linear_sum_assignment` (Hungarian),
3. Computing the sort-based loss,
4. Asserting they are equal to floating-point precision for $\phi \in \{|\cdot|, (\cdot)^2, \text{Huber}\}$.

If this test ever fails, the theorem is wrong and we must stop.

---

## 8. What this theorem buys us (paper-wise)

1. **Scalability claim**: The sort-match loss is the computational hinge that makes literature-scale training feasible. Without it, $O(n^3)$ Hungarian per molecule is a batching nightmare.
2. **Differentiability claim**: Sort is batched, GPU-native, and has well-defined autodiff. Hungarian algorithm does not — people use relaxations (Sinkhorn, OT) which introduce approximation error.
3. **Exactness claim**: Unlike Sinkhorn/OT relaxations, our reduction is **exact**: we are not approximating Hungarian; we are solving a simpler equivalent problem. This is a real technical win over generic set-prediction methods that apply optimal-transport relaxations in the hope of recovering Hungarian behavior.
4. **Mathematical elegance**: The theorem is a three-line application of convex analysis. Reviewers will find this convincing and easy to verify. That matters for Nature CS editorial acceptance — the theoretical piece has to be both novel-enough-to-matter and simple-enough-to-trust.

---

## 9. Open mathematical questions (optional depth)

- **Stability under noisy labels.** If $y_i^\star$ are observed with noise $\epsilon_i$, the optimal sort permutation is an unbiased estimator of the true matching only asymptotically. Finite-sample bounds would be nice but are not essential for the paper.
- **Connection to 1-Wasserstein distance.** For $\phi(t) = |t|$, the sort-match loss equals the 1-Wasserstein distance between the empirical distributions $\hat{\mu} = \frac{1}{n} \sum \delta_{\hat{y}_i}$ and $\mu^\star = \frac{1}{n} \sum \delta_{y_i^\star}$. This gives us a probabilistic interpretation and a link to OT literature that may please chemistry reviewers.
- **Higher-dimensional generalizations.** If we also predicted, say, peak intensities as a second scalar, the cost would be 2D and Theorem 1 would not apply directly. Future work.

---

## 10. Status before moving to code

- ✅ Theorem stated formally
- ✅ Proof self-contained, no hand-waving, no appeal to "obviously"
- ✅ Corollaries for MAE, MSE, Huber stated
- ✅ DETR relationship clarified (they can't do this; we can)
- ✅ Limitations honestly discussed (degeneracy, noise, non-convex $\phi$)
- ✅ Numerical sanity check planned
- ✅ What this buys the paper articulated

**Next step (Phase 3):** implement the sort-match loss in PyTorch with MPS support, and immediately run the numerical sanity check against `scipy.optimize.linear_sum_assignment`. If the test fails, we stop and re-examine the math.
