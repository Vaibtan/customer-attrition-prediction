"""Latent-common-cause simulator: scalar-health population + event generator.

Phase 0 freezes the *functional forms* (``kernels``) + the tuning harness that sets
their coefficients; Phase 1 builds the population/event generator on top; drift
injection is Phase 4. The frozen world is specified in ``docs/simulator_spec.md``
and ``simulator.params.json`` (see ``PHASE0_LOCK_DECISIONS.md`` D5).
"""
