"""Log-only shadow scoring: the @challenger sees live traffic, callers never see it (ADR 0006).

After the champion response is sent, the SAME assembled inputs are scored with whatever model
holds the ``@challenger`` alias and appended to a JSONL shadow log -- paired champion/challenger
predictions on identical live inputs, explicitly framed as future promotion-gate evidence. No
alias set (or any shadow failure) is a silent no-op: shadow must never break or slow serving.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from churn.serving.champion import ModelUnavailable


class ShadowScorer:
    """Score with the challenger resolver and append one JSONL record per request.

    ``resolver`` duck-types the scorer contract (``score`` + ``run_id``), normally a
    :class:`~churn.serving.champion.ChampionResolver` bound to ``@challenger``. ``on_outcome``
    (if given) receives ``"scored" | "no_challenger" | "error"`` -- the API layer wires it to a
    Prometheus counter so shadow activity is observable.
    """

    def __init__(
        self,
        resolver,
        log_path: Path | str,
        on_outcome: Callable[[str], None] | None = None,
    ) -> None:
        self._resolver = resolver
        self._log_path = Path(log_path)
        self._on_outcome = on_outcome or (lambda outcome: None)

    def score_and_log(
        self,
        *,
        customer_id: str,
        as_of: str,
        static: Mapping[str, object],
        event: Mapping[str, object],
        champion_run_id: str | None,
        champion_probability: float,
    ) -> bool:
        """Best-effort shadow score; returns True iff a record was written."""
        try:
            result = self._resolver.score(static, event)
            record = {
                "ts": time.time(),
                "customer_id": customer_id,
                "as_of": as_of,
                "champion_run_id": champion_run_id,
                "champion_probability": champion_probability,
                "challenger_run_id": self._resolver.run_id,
                "challenger_probability": result.churn_probability,
            }
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except ModelUnavailable:
            self._on_outcome("no_challenger")  # no @challenger aliased -- by-design no-op
            return False
        except Exception:  # noqa: BLE001 -- shadow must never propagate into serving
            self._on_outcome("error")
            return False
        self._on_outcome("scored")
        return True
