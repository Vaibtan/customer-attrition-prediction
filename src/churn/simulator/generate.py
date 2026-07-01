"""Dataset orchestrator -- assembles the frozen world into one reproducible ``Dataset``.

Composes the tested pieces (population sampler -> quality index -> latent health paths -> the single
hazard Bernoulli label -> the health-driven event log) into the artifacts Phase 1 measures on:
``customers`` (static + ``q`` + ``h(t0)`` + the analytic ``oracle_score``), the long ``events`` log,
``labels`` (the resimulated ``y`` + the anchor's real label), and the ``(customer_id, t0)`` cohort.

The label is drawn as ``Bernoulli(hazard_prob(h(t0), q))`` -- the analytic Bayes oracle score -- and
is **never** fed back into event generation (invariant i). Everything is a pure function of
``(params, seed)``; the label-shuffle and null-stream controls (D5.8) are exposed here so both hold
health + label fixed while only the relevant stream changes (D6.8 control-validity).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from churn.simulator import events as events_mod
from churn.simulator import kernels as K
from churn.simulator import latent as L
from churn.simulator import params as P
from churn.simulator import population as POP
from churn.simulator import rng as R

# Re-export so callers reach the shared t0 via the orchestrator (tests use G.events.ANCHOR_T0).
events = events_mod

STATIC_FEATURES = ["region", "device_type", "subscription_plan", *P.STATIC_NUMERICS]


@dataclass(frozen=True)
class Dataset:
    """The frozen-world artifacts for one seed: static + label + events + cohort."""

    customers: pd.DataFrame
    labels: pd.DataFrame
    events: pd.DataFrame
    cohort: pd.DataFrame
    params: dict

    def anchor_mask(self) -> np.ndarray:
        return self.labels["is_anchor"].to_numpy(dtype=bool)

    def shuffle_labels(self, rng: np.random.Generator) -> Dataset:
        """Label-shuffle negative control (D5.8): customer-level permutation of the simulated y."""
        labels = self.labels.copy()
        labels["y"] = labels["y"].to_numpy()[rng.permutation(len(labels))]
        return replace(self, labels=labels)


def build_population_dataset(
    params: dict,
    seed: int,
    n_synthetic: int | None = None,
    t0: pd.Timestamp = events_mod.ANCHOR_T0,
    null_stream: bool = False,
) -> Dataset:
    """Build the full seeded ``Dataset`` (population + health + label + events)."""
    streams = R.make_streams(seed)
    pop = POP.sample_population(params, seed, n_synthetic)

    q = P.quality_index(pop[STATIC_FEATURES], params)
    health = L.simulate_health_paths(q, params, streams["latent"])
    h_t0 = L.health_at_t0(health)

    haz = params["hazard"]
    oracle_score = K.hazard_prob(h_t0, q, haz["alpha0"], haz["alpha_h"], haz["alpha_stat"])
    y = (streams["label"].random(len(pop)) < oracle_score).astype(int)

    customers = pop.copy()
    customers["q"] = q
    customers["h_t0"] = h_t0
    customers["oracle_score"] = oracle_score

    labels = pd.DataFrame(
        {
            "customer_id": pop["customer_id"].to_numpy(),
            "t0": t0,
            "y": y,
            "is_anchor": pop["is_anchor"].to_numpy(),
            "real_churned": pop["real_churned"].to_numpy(),
        }
    )
    cohort = pd.DataFrame({"customer_id": pop["customer_id"].to_numpy(), "t0": t0})
    ev = events_mod.generate_events(
        pop["customer_id"].to_numpy(), health, params, streams, t0=t0, null_stream=null_stream
    )
    return Dataset(customers=customers, labels=labels, events=ev, cohort=cohort, params=params)
