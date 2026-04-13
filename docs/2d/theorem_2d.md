# 2D Sort-Match via Sliced Wasserstein

## Setting

A molecule has $n$ carbons with attached hydrogens. A 2D HSQC spectrum is the set of cross-peaks at
$$\mathbf{P}^\star = \{(\delta_{H,i}^\star, \delta_{C,i}^\star) : i = 1, \ldots, n\} \subset \mathbb{R}^2,$$
where cross-peak $i$ reports the direct $^1$H–$^{13}$C one-bond coupling at carbon $i$.

A model produces predicted cross-peaks
$$\hat{\mathbf{P}} = \{(\hat{\delta}_{H,i}, \hat{\delta}_{C,i}) : i = 1, \ldots, n\}.$$

The spectra are unassigned: only the *multi-set* of peaks is observed; the atom identity is lost.

The natural permutation-invariant loss between the predicted and observed peak sets is the 2-Wasserstein distance, defined here as the optimal bipartite matching cost under an L2 cost:
$$W_2^2(\hat{\mathbf{P}}, \mathbf{P}^\star) = \min_{\sigma \in \mathfrak{S}_n} \sum_{i=1}^n \left( (\hat{\delta}_{H,i} - \delta^\star_{H,\sigma(i)})^2 + (\hat{\delta}_{C,i} - \delta^\star_{C,\sigma(i)})^2 \right).$$

The 1D sort-match reduction (Theorem 1 of the main paper) does **not** extend to the 2D case: the optimal 2D matching is generally *not* the product of two independent 1D sorted matchings. A counterexample: predicted $\{(1,10),(2,20)\}$ and target $\{(1,20),(2,10)\}$ — the optimal 2D matching has cost $2$, but the sum of the two 1D sort-match losses is $0$.

## 2D Sort-Match via Sliced Wasserstein

The cleanest generalization is the **sliced Wasserstein distance** (Bonneel, Rabin, Peyré, Pfister, 2015), which reduces 2D optimal transport to an expectation over 1D projections:

$$SW_2^2(\hat{\mathbf{P}}, \mathbf{P}^\star) = \mathbb{E}_{\theta \sim \mathrm{Unif}(S^1)} \left[ W_2^2(\Pi_\theta \hat{\mathbf{P}}, \Pi_\theta \mathbf{P}^\star) \right],$$

where $\Pi_\theta : \mathbb{R}^2 \to \mathbb{R}$, $\mathbf{p} \mapsto \theta^\top \mathbf{p}$ is the 1D orthogonal projection onto the direction $\theta$. Because each projection reduces the problem to 1D, **Theorem 1 of the main paper applies to each projection**: the inner $W_2^2$ is computed by sorting both projected point sets and summing squared differences in order.

A Monte-Carlo estimator with $K$ directions is
$$\widehat{SW_2^2}(\hat{\mathbf{P}}, \mathbf{P}^\star; \{\theta_k\}_{k=1}^K) = \frac{1}{K} \sum_{k=1}^K \mathcal{L}_{\mathrm{sort}}(\Pi_{\theta_k} \hat{\mathbf{P}}, \Pi_{\theta_k} \mathbf{P}^\star),$$
computable in $O(K n \log n)$ per molecule.

## Theorem 2 (Sliced Sort-Match)

> **Theorem 2.** *Let $\hat{\mathbf{P}}, \mathbf{P}^\star \in \mathbb{R}^{n \times 2}$ be two point sets of cardinality $n$. For any set of unit directions $\{\theta_k\}_{k=1}^K \subset S^1$ and a convex per-coordinate cost $\phi$, the sliced sort-match loss*
> $$\mathcal{L}_{\mathrm{SSW}}(\hat{\mathbf{P}}, \mathbf{P}^\star) = \frac{1}{K} \sum_{k=1}^K \sum_{i=1}^n \phi\!\left( (\Pi_{\theta_k}\hat{\mathbf{P}})_{\uparrow, i} - (\Pi_{\theta_k}\mathbf{P}^\star)_{\uparrow, i} \right)$$
> *is* (i) *permutation-invariant in both arguments,* (ii) *non-negative,* (iii) *zero if and only if $\hat{\mathbf{P}} = \mathbf{P}^\star$ as multisets when $K$ spans $\mathbb{R}^2$,* (iv) *differentiable almost everywhere via $K$ parallel $O(n \log n)$ sort operations, and* (v) *an unbiased estimator of $SW_2^2$ (up to constant) when $\{\theta_k\}$ are drawn iid from $\mathrm{Unif}(S^1)$.*

**Proof sketch.** (i)-(ii) are immediate from the 1D sort-match lemma applied to each projection. (iii) follows because if $\hat{\mathbf{P}} \neq \mathbf{P}^\star$ as multisets then there exists a direction $\theta$ along which they have different empirical distributions, yielding non-zero 1D sort-match loss; if $K$ spans $\mathbb{R}^2$ this direction is represented. (iv) is the standard property of `torch.sort` differentiability. (v) is the defining property of sliced Wasserstein. $\quad\square$

## Axis-aligned special case

When the two spectral axes are the natural ¹H and ¹³C dimensions and we use exactly $K = 2$ directions aligned with the axes ($\theta_1 = (1, 0)$, $\theta_2 = (0, 1)$), the sliced sort-match collapses to
$$\mathcal{L}_{\mathrm{axis}}(\hat{\mathbf{P}}, \mathbf{P}^\star) = \mathcal{L}_{\mathrm{sort}}(\hat{\boldsymbol{\delta}}_H, \boldsymbol{\delta}_H^\star) + \mathcal{L}_{\mathrm{sort}}(\hat{\boldsymbol{\delta}}_C, \boldsymbol{\delta}_C^\star),$$
the sum of two independent 1D sort-match losses along the two nuclei axes. This is the cheapest possible 2D loss — zero random-direction overhead — but it is a biased estimator: it can be *zero* when the true $W_2^2$ is positive (if the two predicted and target empirical marginals match but the joint does not). For training it is useful as a warm-start regularizer; for the main loss, we prefer the random-projection version with $K \geq 4$.

## Why this matters for NMR SSL

Sliced sort-match provides a differentiable, permutation-invariant, $O(K n \log n)$ loss for matching predicted to observed 2D peak sets. This unlocks a new class of supervision signals:

1. **Unassigned 2D HSQC peak lists** — abundant in literature but unused by existing ML methods because of the assignment ambiguity — can be consumed as SSL supervision through the sliced sort-match loss.
2. **Richer per-molecule information**: an unassigned 2D HSQC peak list constrains both the ¹H and the ¹³C shift distribution simultaneously, whereas an unassigned 1D peak list constrains only one nucleus. Per-molecule information content is roughly doubled.
3. **Architecture-agnostic**: the loss plugs into any graph neural network that outputs per-atom ¹H and ¹³C shifts.

The core hypothesis of the 2D experiment: *at matched label budgets, training with unassigned 2D HSQC supervision yields lower 1D ¹³C test MAE than training with unassigned 1D ¹³C supervision.*

## Implementation notes

- We use $K = 8$ Gaussian random directions drawn fresh at every batch, renormalized to unit length.
- The per-direction loss is the masked 1D sort-match already implemented in `src/losses.py::masked_sort_match_loss`.
- Gradients flow back through the projection and through `torch.sort` via gather-scatter.
- For the axis-aligned special case we include a second pathway that uses $K = 2$ fixed directions — useful for ablation.
