"""Process-global RNG seeding, and a record of what was applied.

The record exists because a seed cannot be recovered after the fact. Neither ``random`` nor
``numpy`` exposes a getter for the value passed to ``seed()`` - ``getstate()`` and
``get_state()`` return Mersenne Twister state, not a seed - and ``torch.initial_seed()``
returns a value torch generated itself when nothing was set. So the only way to log a truthful
seed is to record it at the moment it is applied.

Seeding the main process is enough to make the whole pipeline reproducible: each DataLoader
worker seeds random, numpy and torch from ``base_seed + worker_id``, and ``base_seed`` is drawn
from the global torch generator that ``torch.manual_seed`` controls.
"""
import logging
import random
import secrets

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: numpy's legacy global RandomState rejects anything above this, which makes it the binding
#: constraint on every seed we accept or generate.
MAX_SEED = 2 ** 32 - 1

_applied = None


def apply_seed(seed: int | None = None) -> dict:
    """Seed random, numpy and torch, and return what was applied.

    ``seed=None`` draws one from OS entropy, so a run is reproducible either way: the generated
    value is recorded and logged, and can be pasted back into the config to replay the run.

    Must be called before the model is constructed - weight initialization draws from the torch
    RNG, so seeding any later leaves the initial weights uncontrolled.
    """
    global _applied

    source = 'config'
    if seed is None:
        seed = secrets.randbits(32)
        source = 'generated'

    seed = int(seed)
    if not 0 <= seed <= MAX_SEED:
        # numpy raises on its own, but its message names neither the config key nor the caller.
        raise ValueError(f"Seed must be in [0, {MAX_SEED}], got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device, via torch.cuda.manual_seed_all

    # Stored per library even though the values match, so the record's shape already matches the
    # three params that get logged and the two can diverge later without a format change.
    _applied = {'random': seed, 'numpy': seed, 'torch': seed, 'source': source}
    logger.info(f"Seeded random, numpy and torch with {seed} ({source})")
    return dict(_applied)


def get_applied_seeds() -> dict | None:
    """What apply_seed last applied, or None if it was never called."""
    return dict(_applied) if _applied is not None else None
