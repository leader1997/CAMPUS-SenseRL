"""Reproducibility: same seed yields identical random masks."""

from __future__ import annotations

import numpy as np

from campus_senserl.models.baselines import make_random_mask
from campus_senserl.models.graph_reconstruction import apply_mask_scheme
from campus_senserl.utils import set_seed


def test_apply_mask_scheme_same_seed_same_mask():
    observed = np.random.default_rng(0).random((30, 10)) < 0.85
    m1 = apply_mask_scheme(observed, "random", rate=0.25, seed=42)
    m2 = apply_mask_scheme(observed, "random", rate=0.25, seed=42)
    np.testing.assert_array_equal(m1, m2)


def test_apply_mask_scheme_different_seed_different_mask():
    observed = np.ones((20, 5), dtype=bool)
    m1 = apply_mask_scheme(observed, "random", rate=0.4, seed=1)
    m2 = apply_mask_scheme(observed, "random", rate=0.4, seed=2)
    assert not np.array_equal(m1, m2)


def test_make_random_mask_reproducible():
    import pandas as pd

    rng = np.random.default_rng(7)
    wide = pd.DataFrame(rng.normal(size=(15, 6)))
    m1 = make_random_mask(wide, rate=0.3, seed=99)
    m2 = make_random_mask(wide, rate=0.3, seed=99)
    pd.testing.assert_frame_equal(m1, m2)


def test_set_seed_stabilizes_numpy():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    np.testing.assert_allclose(a, b)
