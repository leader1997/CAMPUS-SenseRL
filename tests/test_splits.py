"""Tests for chronological split ordering — no shuffle leakage."""

from __future__ import annotations

import pandas as pd
import pytest

from campus_senserl.data.splits import (
    assign_split,
    chronological_split_bounds,
    rolling_origin_folds,
    split_frame,
)


def test_chronological_bounds_are_monotonic():
    times = pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC")
    bounds = chronological_split_bounds(times, train_frac=0.6, val_frac=0.2, test_frac=0.2)
    assert bounds["t_min"] <= bounds["train_end"] <= bounds["val_end"] <= bounds["test_end"]


def test_assign_split_respects_time_order():
    times = pd.date_range("2024-01-01", periods=50, freq="15min", tz="UTC")
    bounds = chronological_split_bounds(times)
    splits = assign_split(times, bounds)
    df = pd.DataFrame({"slot": times, "split": splits})
    for split_name in ["train", "val", "test"]:
        subset = df[df["split"] == split_name]
        if len(subset) > 1:
            assert subset["slot"].is_monotonic_increasing

    train_max = df.loc[df["split"] == "train", "slot"].max()
    val_max = df.loc[df["split"] == "val", "slot"].max()
    val_min = df.loc[df["split"] == "val", "slot"].min()
    test_min = df.loc[df["split"] == "test", "slot"].min()
    assert train_max <= val_min
    assert val_max <= test_min or df["split"].eq("test").sum() == 0


def test_no_future_slot_in_train():
    times = pd.date_range("2024-06-01", periods=200, freq="15min", tz="UTC")
    bounds = chronological_split_bounds(times)
    splits = assign_split(times, bounds)
    df = pd.DataFrame({"slot": times, "split": splits})
    train_end = bounds["train_end"]
    assert df.loc[df["split"] == "train", "slot"].max() <= pd.Timestamp(train_end).tz_convert("UTC")


def test_split_frame_partitions_without_shuffle():
    times = pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")
    bounds = chronological_split_bounds(times)
    df = pd.DataFrame(
        {
            "slot": times.repeat(3),
            "deveui": ["A", "B", "C"] * len(times),
            "split": assign_split(times.repeat(3), bounds),
        }
    )
    parts = split_frame(df)
    assert set(parts.keys()) == {"train", "val", "test"}
    assert sum(len(v) for v in parts.values()) == len(df)
    recombined = pd.concat(parts.values()).sort_index()
    pd.testing.assert_frame_equal(recombined.sort_values(["slot", "deveui"]).reset_index(drop=True), df.sort_values(["slot", "deveui"]).reset_index(drop=True))


def test_rolling_origin_folds_expand_chronologically():
    times = pd.date_range("2024-01-01", periods=80, freq="15min", tz="UTC")
    folds = rolling_origin_folds(times, n_folds=3, min_train_frac=0.5)
    assert len(folds) >= 1
    prev_train_end = pd.Timestamp.min.tz_localize("UTC")
    for fold in folds:
        assert fold["train_end"] >= prev_train_end
        assert fold["val_end"] >= fold["train_end"]
        assert fold["test_end"] >= fold["val_end"]
        prev_train_end = fold["train_end"]


def test_fractions_must_sum_to_one():
    times = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    with pytest.raises(AssertionError):
        chronological_split_bounds(times, train_frac=0.5, val_frac=0.3, test_frac=0.1)
