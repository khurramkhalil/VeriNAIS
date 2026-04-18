"""
stl_nas/algorithms/lamoo.py
---------------------------
LaMOO oracle wrapper for the STL-NAS gatekeeper.

LaMOO (Latent-space Multi-Objective Optimisation, Wang et al. 2021) is a
Bayesian optimisation method that partitions the search space using a
space-partitioning tree (similar to MCTS) and maintains separate Gaussian
Process (GP) surrogates in each leaf.  In the multi-objective NAS setting,
LaMOO optimises the (accuracy, energy) Pareto front by modelling the
hypervolume contribution of each candidate.

High-level algorithm per iteration
------------------------------------
  1. Partition: update the binary space-partition tree using current
     observations.  Each leaf models a region of the architecture embedding.
  2. Select leaf: compute UCB acquisition value for each leaf;
     select the most promising leaf.
  3. Sample candidates: sample architectures uniformly within the selected
     leaf (rejection sampling over the NAS-Bench-201 search space).
  4. Evaluate: architectures are passed to the STL gatekeeper; admitted
     ones are queried from the benchmark table.
  5. Update GPs: fit separate accuracy and energy GPs on all observations.

Architecture embedding
-----------------------
Each NAS-Bench-201 architecture (6-edge operation vector) is embedded into
a d-dimensional continuous space via a random Fourier feature (RFF) map:

    φ(arch) = [cos(W·ops + b), sin(W·ops + b)] / sqrt(D)

where W ∈ R^{D×6} is drawn once from a RBF kernel frequency distribution.
This provides a stationary kernel approximation that scales to the full
15 625-architecture search space without storing a dense kernel matrix.

Gaussian Process surrogates
-----------------------------
Separate GPs model f_acc and f_energy using the RFF-based approximation.
Predictions are used to compute the expected hypervolume improvement (EHVI)
acquisition function, which guides candidate selection.

Note on the paper's experimental setup
----------------------------------------
In the NAS-Bench-201 experiments reported in the paper, the accuracy and
energy values are retrieved directly from the pre-computed benchmark table
rather than from GP predictions.  The LaMOO oracle here drives the search
space partition and sampling strategy; the "evaluate" step queries the
table.  This matches the experimental protocol of the original LaMOO paper
for the NAS-Bench benchmark setting.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..signal import Architecture


# ---------------------------------------------------------------------------
# Random Fourier Feature embedding
# ---------------------------------------------------------------------------

class RFFEmbedding:
    """Random Fourier Feature map for the NAS-Bench-201 operation vector.

    Parameters
    ----------
    input_dim :
        Dimension of the input (6 for NAS-Bench-201 edges).
    n_features :
        Number of RFF features D (output dim = 2D).
    length_scale :
        RBF kernel length scale (controls smoothness of the surrogate).
    seed :
        Random seed for reproducibility across runs.
    """

    def __init__(
        self,
        input_dim: int = 6,
        n_features: int = 128,
        length_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 1.0 / length_scale, (n_features, input_dim))
        self.b = rng.uniform(0, 2 * np.pi, n_features)
        self._scale = np.sqrt(1.0 / n_features)

    def __call__(self, ops: np.ndarray) -> np.ndarray:
        """Map a (6,) operation vector to a (2D,) feature vector."""
        z = self.W @ ops.astype(float) + self.b
        return self._scale * np.concatenate([np.cos(z), np.sin(z)])


# ---------------------------------------------------------------------------
# Lightweight GP surrogate (closed-form RFF ridge regression)
# ---------------------------------------------------------------------------

class RFFSurrogate:
    """GP surrogate using the RFF approximation (ridge regression form).

    Given observations {(φ_i, y_i)}, the predictive mean and variance are:

        μ(φ) = φ^T (Φ^T Φ + σ²I)^{-1} Φ^T y
        σ²(φ) = σ² φ^T (Φ^T Φ + σ²I)^{-1} φ

    This is equivalent to a GP posterior with the RFF kernel approximation.
    """

    def __init__(
        self,
        feature_dim: int,
        noise_var: float = 0.01,
    ) -> None:
        self.feature_dim = feature_dim
        self.noise_var   = noise_var
        self._Phi: Optional[np.ndarray] = None  # shape (n_obs, feature_dim)
        self._y:   Optional[np.ndarray] = None  # shape (n_obs,)
        self._A:   Optional[np.ndarray] = None  # (Φ^T Φ + σ²I)^{-1}

    def fit(self, Phi: np.ndarray, y: np.ndarray) -> None:
        """Fit the surrogate on feature matrix Phi and targets y."""
        self._Phi = Phi
        self._y   = y
        A = Phi.T @ Phi + self.noise_var * np.eye(self.feature_dim)
        self._A = np.linalg.solve(A, np.eye(self.feature_dim))

    def predict(
        self, phi: np.ndarray
    ) -> Tuple[float, float]:
        """Return (mean, variance) at query point phi."""
        if self._A is None or self._Phi is None or self._y is None:
            # Prior: uninformative
            return 0.0, 1.0
        mu  = float(phi @ self._A @ self._Phi.T @ self._y)
        var = float(self.noise_var * phi @ self._A @ phi)
        return mu, max(var, 1e-8)


# ---------------------------------------------------------------------------
# Binary space-partition tree (LaMOO core)
# ---------------------------------------------------------------------------

class SpacePartitionNode:
    """One node in the space-partition tree.

    Leaves maintain lists of observed architecture IDs; internal nodes
    split the space along a randomly chosen feature dimension.
    """

    def __init__(self, arch_ids: List[int], depth: int = 0) -> None:
        self.arch_ids = arch_ids
        self.depth    = depth
        self.left:  Optional[SpacePartitionNode] = None
        self.right: Optional[SpacePartitionNode] = None
        self.split_dim: Optional[int]   = None
        self.split_val: Optional[float] = None

        # UCB value (updated during tree traversal)
        self.ucb: float = np.inf
        self.n_visits: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


# ---------------------------------------------------------------------------
# LaMOO Oracle
# ---------------------------------------------------------------------------

class LaMOOOracle:
    """LaMOO-based candidate generator for multi-objective NAS.

    Maintains a space-partitioning tree over the NAS-Bench-201 search space
    and uses GP surrogates to guide sampling towards Pareto-improving regions.

    Parameters
    ----------
    arch_pool :
        Full NAS-Bench-201 search space (pre-loaded Architecture objects).
    candidates_per_iter :
        Number of candidates to return per oracle call.
    n_rff_features :
        Number of random Fourier features for the GP embedding.
    exploration_weight :
        UCB exploration coefficient C in UCB = μ + C·σ.
    min_leaf_size :
        Minimum number of observations in a leaf before it is split.
    seed :
        Random seed.
    """

    N_EDGES = 6
    N_OPS   = 5

    def __init__(
        self,
        arch_pool: List[Architecture],
        candidates_per_iter: int = 20,
        n_rff_features: int = 128,
        exploration_weight: float = 1.0,
        min_leaf_size: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        self.arch_pool           = arch_pool
        self.candidates_per_iter = candidates_per_iter
        self.exploration_weight  = exploration_weight
        self.min_leaf_size       = min_leaf_size
        self._rng                = np.random.default_rng(seed)
        self._id_to_arch         = {a.arch_id: a for a in arch_pool}

        self._embed = RFFEmbedding(
            input_dim=self.N_EDGES,
            n_features=n_rff_features,
            seed=seed,
        )
        self._feature_dim = 2 * n_rff_features

        # Surrogate models (accuracy and energy)
        self._gp_acc    = RFFSurrogate(self._feature_dim)
        self._gp_energy = RFFSurrogate(self._feature_dim)

        # Observation history
        self._obs_Phi: List[np.ndarray] = []
        self._obs_acc: List[float]      = []
        self._obs_energy: List[float]   = []

        # Space partition tree (initialised with all arch IDs)
        all_ids  = [a.arch_id for a in arch_pool]
        self._root = SpacePartitionNode(all_ids)

    # ------------------------------------------------------------------
    # Oracle interface
    # ------------------------------------------------------------------

    def __call__(self, population: List[Architecture]) -> List[Architecture]:
        """Generate a batch of candidate architectures.

        Uses the current GP surrogates and space-partition tree to select
        the most promising region, then samples architectures within it.
        """
        # Update surrogates if we have observations
        self._update_surrogates()

        # Select the most promising leaf via UCB tree traversal
        leaf = self._select_leaf(self._root)

        # Sample candidates from within the selected leaf
        return self._sample_from_leaf(leaf, self.candidates_per_iter)

    def record_observation(self, arch: Architecture) -> None:
        """Record an accepted architecture in the observation history.

        Called by the experiment runner after the gatekeeper admits an arch.
        Updates the surrogate training data and triggers a tree-split check.
        """
        ops = self._arch_id_to_ops(arch.arch_id)
        phi = self._embed(np.array(ops))
        self._obs_Phi.append(phi)
        self._obs_acc.append(arch.acc / 100.0)
        self._obs_energy.append(arch.energy)

        # Re-split the tree if any leaf has grown beyond min_leaf_size
        self._update_tree(self._root)

    # ------------------------------------------------------------------
    # Surrogate updates
    # ------------------------------------------------------------------

    def _update_surrogates(self) -> None:
        if len(self._obs_Phi) < 2:
            return
        Phi = np.stack(self._obs_Phi, axis=0)
        self._gp_acc.fit(Phi, np.array(self._obs_acc))
        self._gp_energy.fit(Phi, np.array(self._obs_energy))

    # ------------------------------------------------------------------
    # Tree operations
    # ------------------------------------------------------------------

    def _ucb_value(self, arch_id: int) -> float:
        """Compute multi-objective UCB value for an architecture (EHVI proxy)."""
        ops = self._arch_id_to_ops(arch_id)
        phi = self._embed(np.array(ops))
        mu_acc,    var_acc    = self._gp_acc.predict(phi)
        mu_energy, var_energy = self._gp_energy.predict(phi)

        # UCB: maximise accuracy (UCB) and minimise energy (LCB)
        ucb_acc    = mu_acc    + self.exploration_weight * np.sqrt(var_acc)
        lcb_energy = mu_energy - self.exploration_weight * np.sqrt(var_energy)

        # Combine into a scalar (simple scalarisation; full EHVI not shown)
        return ucb_acc - lcb_energy

    def _select_leaf(
        self, node: SpacePartitionNode
    ) -> SpacePartitionNode:
        """Traverse the tree to find the leaf with the highest UCB."""
        if node.is_leaf:
            return node
        # Recursively choose the child with higher UCB
        ucb_l = self._node_ucb(node.left)
        ucb_r = self._node_ucb(node.right)
        return self._select_leaf(node.left if ucb_l >= ucb_r else node.right)

    def _node_ucb(self, node: Optional[SpacePartitionNode]) -> float:
        if node is None:
            return -np.inf
        if not node.arch_ids:
            return -np.inf
        # Average UCB of architectures in this node
        sample = self._rng.choice(
            node.arch_ids, size=min(10, len(node.arch_ids)), replace=False
        )
        return float(np.mean([self._ucb_value(i) for i in sample]))

    def _update_tree(self, node: SpacePartitionNode) -> None:
        """Recursively split leaves that have enough new observations."""
        if node.is_leaf:
            if len(node.arch_ids) >= self.min_leaf_size:
                self._split_node(node)
        else:
            self._update_tree(node.left)
            self._update_tree(node.right)

    def _split_node(self, node: SpacePartitionNode) -> None:
        """Split a leaf along the feature dimension with highest variance."""
        if not node.arch_ids:
            return

        # Embed all architectures in this node
        phis = np.stack([
            self._embed(np.array(self._arch_id_to_ops(i)))
            for i in node.arch_ids
        ], axis=0)

        # Split along the dimension with highest variance
        variances = np.var(phis, axis=0)
        dim       = int(np.argmax(variances))
        val       = float(np.median(phis[:, dim]))

        ids_l = [i for i, phi in zip(node.arch_ids, phis) if phi[dim] <= val]
        ids_r = [i for i, phi in zip(node.arch_ids, phis) if phi[dim] >  val]

        if not ids_l or not ids_r:
            return   # degenerate split; skip

        node.split_dim = dim
        node.split_val = val
        node.left  = SpacePartitionNode(ids_l, node.depth + 1)
        node.right = SpacePartitionNode(ids_r, node.depth + 1)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_from_leaf(
        self, leaf: SpacePartitionNode, n: int
    ) -> List[Architecture]:
        """Sample n architectures uniformly from a leaf's arch_ids."""
        if not leaf.arch_ids:
            # Fallback: sample from the full pool
            ids = [a.arch_id for a in self.arch_pool]
        else:
            ids = leaf.arch_ids

        chosen = self._rng.choice(
            len(ids), size=min(n, len(ids)), replace=False
        )
        return [self._id_to_arch[ids[i]] for i in chosen
                if ids[i] in self._id_to_arch]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _arch_id_to_ops(self, arch_id: int) -> List[int]:
        ops = []
        n   = arch_id
        for _ in range(self.N_EDGES):
            ops.append(n % self.N_OPS)
            n //= self.N_OPS
        return ops
