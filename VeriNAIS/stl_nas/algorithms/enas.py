"""
stl_nas/algorithms/enas.py
--------------------------
ENAS oracle wrapper for the STL-NAS gatekeeper.

ENAS (Efficient Neural Architecture Search, Pham et al. 2018) uses a
recurrent controller network (LSTM) that samples child architectures and
is trained via REINFORCE to maximise expected accuracy on a validation set.

In STL-NAS, the ENAS controller is used as a black-box oracle O that
samples child architectures one at a time.  The gatekeeper intercepts
each sample BEFORE the expensive accuracy evaluation is performed on the
benchmark, thereby saving compute by rejecting infeasible candidates early.

Accuracy feedback loop
-----------------------
In the full ENAS setting, the controller is updated online using the
validation accuracy of accepted architectures as the reward signal.
The update uses the REINFORCE policy gradient estimator with an
exponential-moving-average baseline to reduce variance:

    ∇_θ J(θ) = Σ_t  (R_t − b_t) · ∇_θ log π_θ(a_t | s_t)

where:
  R_t   = validation accuracy of the sampled architecture (reward)
  b_t   = exponential moving average of past rewards (baseline)
  π_θ   = ENAS controller policy (LSTM)
  a_t   = sampled operation at edge t
  s_t   = controller hidden state at step t

In the NAS-Bench-201 experimental setting used in the paper, the reward
is retrieved from the pre-computed benchmark table (no actual training).

Controller architecture
-----------------------
The controller is a single-layer LSTM that sequentially samples one
operation per edge of the NAS-Bench-201 cell (6 edges × 5 operations).
The hidden state is passed between edges; the output at each step is
fed through a softmax to produce the operation probability distribution.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..signal import Architecture


# ---------------------------------------------------------------------------
# LSTM Controller (minimal implementation for the NAS sampling loop)
# ---------------------------------------------------------------------------

class LSTMController:
    """Lightweight LSTM controller for the ENAS architecture sampler.

    The controller maintains hidden (h) and cell (c) states and produces
    a probability distribution over operations at each edge.

    Parameters
    ----------
    n_ops :
        Number of available operations (5 for NAS-Bench-201).
    n_edges :
        Number of edges in the cell (6 for NAS-Bench-201).
    hidden_size :
        LSTM hidden dimension.
    temperature :
        Softmax temperature for the operation distribution.
        Lower → more peaked (exploitative); higher → more uniform (exploratory).
    seed :
        Random seed.
    """

    N_OPS   = 5
    N_EDGES = 6

    def __init__(
        self,
        n_ops: int = 5,
        n_edges: int = 6,
        hidden_size: int = 64,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.n_ops       = n_ops
        self.n_edges     = n_edges
        self.hidden_size = hidden_size
        self.temperature = temperature
        self._rng        = np.random.default_rng(seed)

        # Xavier-initialised weight matrices
        scale = np.sqrt(2.0 / (hidden_size + hidden_size))
        self.W_ih = self._rng.normal(0, scale, (4 * hidden_size, n_ops))
        self.W_hh = self._rng.normal(0, scale, (4 * hidden_size, hidden_size))
        self.b    = np.zeros(4 * hidden_size)

        # Output embedding (hidden → op logits)
        self.W_out = self._rng.normal(0, scale, (n_ops, hidden_size))
        self.b_out = np.zeros(n_ops)

        # Controller state (reset per episode)
        self.h = np.zeros(hidden_size)
        self.c = np.zeros(hidden_size)

        # Policy gradient accumulators
        self.log_probs: List[float] = []
        self.baseline: float = 0.5    # EMA baseline initialised at 50 % acc
        self.baseline_decay: float = 0.95
        self.lr: float = 0.001        # REINFORCE learning rate

    # ------------------------------------------------------------------
    # LSTM step
    # ------------------------------------------------------------------

    def _lstm_step(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """One LSTM forward step.

        Parameters
        ----------
        x :
            Input vector (one-hot encoded operation, length n_ops).

        Returns
        -------
        h_new, c_new
        """
        gates = self.W_ih @ x + self.W_hh @ self.h + self.b
        H     = self.hidden_size

        i = self._sigmoid(gates[:H])
        f = self._sigmoid(gates[H:2*H])
        g = np.tanh(gates[2*H:3*H])
        o = self._sigmoid(gates[3*H:])

        c_new = f * self.c + i * g
        h_new = o * np.tanh(c_new)
        return h_new, c_new

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset LSTM hidden state at the start of a new episode."""
        self.h = np.zeros(self.hidden_size)
        self.c = np.zeros(self.hidden_size)
        self.log_probs = []

    def sample_architecture(self) -> Tuple[List[int], float]:
        """Sample one architecture by rolling out the LSTM controller.

        Returns
        -------
        ops :
            List of operation indices (length n_edges).
        log_prob :
            Sum of log-probabilities of the sampled operations (used for
            REINFORCE gradient estimation).
        """
        self.reset_state()
        ops: List[int] = []
        log_prob = 0.0

        # Dummy input for the first step
        x = np.zeros(self.n_ops)

        for _ in range(self.n_edges):
            self.h, self.c = self._lstm_step(x)

            # Compute logits and softmax probabilities
            logits = self.W_out @ self.h + self.b_out
            logits /= self.temperature
            logits -= logits.max()  # numerical stability
            probs  = np.exp(logits)
            probs /= probs.sum()

            # Sample operation
            op = int(self._rng.choice(self.n_ops, p=probs))
            ops.append(op)
            log_prob += np.log(probs[op] + 1e-8)

            # Next input: one-hot of sampled operation
            x = np.zeros(self.n_ops)
            x[op] = 1.0

        self.log_probs.append(log_prob)
        return ops, log_prob

    # ------------------------------------------------------------------
    # REINFORCE update
    # ------------------------------------------------------------------

    def update(self, reward: float, log_prob: float) -> None:
        """Update controller weights using the REINFORCE gradient estimator.

        Parameters
        ----------
        reward :
            Validation accuracy of the accepted architecture (normalised to
            [0, 1]).  Retrieved from the NAS-Bench-201 table in our setting.
        log_prob :
            Log-probability of the sampled architecture under the current policy.
        """
        # Update baseline (exponential moving average)
        self.baseline = (
            self.baseline_decay * self.baseline
            + (1 - self.baseline_decay) * reward
        )

        advantage = reward - self.baseline
        grad_scale = self.lr * advantage

        # Approximate gradient step on output weights
        # (full backprop through the LSTM is omitted for brevity; in practice
        # use PyTorch autograd)
        self.W_out += grad_scale * np.outer(
            np.ones(self.n_ops), self.h
        )


# ---------------------------------------------------------------------------
# ENAS Oracle
# ---------------------------------------------------------------------------

class ENASOracle:
    """ENAS oracle: samples child architectures via an LSTM controller.

    The gatekeeper calls this oracle at each iteration to obtain a batch
    of candidate architectures.  Each sample is generated independently
    by rolling out the LSTM.

    After the gatekeeper admits a candidate, the oracle's update() method
    should be called with the reward (accuracy from the benchmark table) to
    update the controller policy.

    Parameters
    ----------
    arch_pool :
        Full NAS-Bench-201 search space (pre-loaded Architecture objects).
    candidates_per_iter :
        Number of architectures to sample per call.
    hidden_size :
        LSTM hidden size.
    temperature :
        Softmax temperature.  Annealed during training (see update()).
    seed :
        Random seed.
    """

    N_OPS   = 5
    N_EDGES = 6

    def __init__(
        self,
        arch_pool: List[Architecture],
        candidates_per_iter: int = 20,
        hidden_size: int = 64,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.arch_pool           = arch_pool
        self.candidates_per_iter = candidates_per_iter
        self._id_to_arch         = {a.arch_id: a for a in arch_pool}
        self._rng                = np.random.default_rng(seed)

        self.controller = LSTMController(
            n_ops=self.N_OPS,
            n_edges=self.N_EDGES,
            hidden_size=hidden_size,
            temperature=temperature,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Oracle interface
    # ------------------------------------------------------------------

    def __call__(self, population: List[Architecture]) -> List[Architecture]:
        """Sample a batch of candidate architectures.

        The current population is not used directly by ENAS (the controller
        is stateful), but is accepted for interface compatibility with the
        gatekeeper.
        """
        candidates: List[Architecture] = []
        for _ in range(self.candidates_per_iter):
            ops, _ = self.controller.sample_architecture()
            arch_id = self._ops_to_arch_id(ops)
            if arch_id in self._id_to_arch:
                candidates.append(self._id_to_arch[arch_id])
        return candidates

    def update_reward(self, arch: Architecture) -> None:
        """Update controller with the reward from an accepted architecture.

        Called by the experiment runner after the gatekeeper admits the arch.
        Reward = normalised accuracy (acc / 100.0).
        """
        reward    = arch.acc / 100.0
        log_prob  = self.controller.log_probs[-1] if self.controller.log_probs else 0.0
        self.controller.update(reward, log_prob)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ops_to_arch_id(self, ops: List[int]) -> int:
        arch_id = 0
        for i, op in enumerate(ops):
            arch_id += op * (self.N_OPS ** i)
        return arch_id
