# -*- coding: utf-8 -*-
"""
Robust patches for GluonTS/MG-TSD interop.

- Patch LSTM.forward: auto dtype alignment + auto pad/trim feature dim to match input_size
- Patch Predictor: ensure freq/frequency
- Patch ProcessStartField & split helper: keep Period / fallback freq
- Patch AddTimeFeatures:
    * Never return 0 columns (fallback to ones)
    * Fast path for BusinessDay-like freqs (avoid heavy pandas.date_range)
    * LRU cache for other freqs
    * Do not touch Timestamp.freq anywhere (avoid deprecation)

"""

from __future__ import annotations

import inspect
import sys
from typing import Optional
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

# silence pandas deprecation about Timestamp.freq
warnings.filterwarnings("ignore", message=".*Timestamp.freq is deprecated.*")


def _print_once(msg: str):
    if msg not in _print_once._seen:
        _print_once._seen.add(msg)
        print(msg, file=sys.stderr)
_print_once._seen = set()


# ---------- utils ----------
def _to_ts_start(start) -> pd.Timestamp:
    if isinstance(start, pd.Period):
        try:
            return start.to_timestamp(how="start").tz_localize(None)
        except Exception:
            return pd.Timestamp(start.start_time).tz_localize(None)
    ts = pd.Timestamp(start)
    try:
        return ts.tz_localize(None)
    except Exception:
        return ts


def _safe_int_len(target) -> int:
    arr = np.asarray(target)
    if arr.ndim == 1:
        return int(arr.shape[0])
    if arr.ndim >= 2:
        return int(arr.shape[-1])
    return int(len(target))


def _vstack_or_ones(rows, length: int) -> np.ndarray:
    if not rows:
        return np.ones((1, int(length)), dtype=np.float32)
    out = np.vstack(rows)
    if out.size == 0 or out.shape[-1] != int(length):
        return np.ones((1, int(length)), dtype=np.float32)
    return out.astype(np.float32, copy=False)


def _ndim2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


# ---------- patches ----------
def _patch_predictor_freqkw(default_freq_str: str):
    def _patch_cls(cls):
        try:
            init = getattr(cls, "__init__")
            if getattr(init, "_freq_patched", False):
                return False
            orig = init
            sig = inspect.signature(orig)
            names = set(sig.parameters.keys())
            wants_freq = "freq" in names
            wants_frequency = "frequency" in names

            def new_init(self, *args, **kwargs):
                if "freq" in kwargs and not wants_freq and wants_frequency and "frequency" not in kwargs:
                    kwargs["frequency"] = kwargs.pop("freq")
                if "frequency" in kwargs and not wants_frequency and wants_freq and "freq" not in kwargs:
                    kwargs["freq"] = kwargs.pop("frequency")
                if wants_freq and "freq" not in kwargs:
                    kwargs["freq"] = default_freq_str
                if wants_frequency and "frequency" not in kwargs:
                    kwargs["frequency"] = default_freq_str
                if not wants_freq and not wants_frequency:
                    kwargs.pop("freq", None)
                    kwargs.pop("frequency", None)
                return orig(self, *args, **kwargs)

            new_init._freq_patched = True
            cls.__init__ = new_init
            _print_once(f"[patch] estimator.PyTorchPredictor patched: ensure freq/frequency (default={default_freq_str})")
            return True
        except Exception:
            return False

    done = False
    for dotted in ("estimator", "mgtsd_estimator", "gluonts.torch.model.predictor"):
        try:
            mod = __import__(dotted, fromlist=["*"])
            if hasattr(mod, "PyTorchPredictor"):
                done |= _patch_cls(mod.PyTorchPredictor)
        except Exception:
            pass
    if not done:
        _print_once("[patch] no PyTorchPredictor needed patching (already compatible)")


def _patch_process_start_field_keep_period(default_freq_str: str):
    import gluonts.dataset.common as gdc
    if getattr(gdc.ProcessStartField, "_keep_period_patched", False):
        return
    off = to_offset(default_freq_str)

    def _proc(timestamp_input, freq):
        try:
            use_off = to_offset(freq) if freq is not None else off
        except Exception:
            use_off = off
        if isinstance(timestamp_input, pd.Period):
            try:
                _ = timestamp_input.freqstr
                return timestamp_input
            except Exception:
                return pd.Period(timestamp_input.start_time, freq=use_off)
        ts = pd.Timestamp(timestamp_input)
        return pd.Period(ts, freq=use_off)

    gdc.ProcessStartField.process = staticmethod(_proc)
    gdc.ProcessStartField._keep_period_patched = True
    _print_once("[patch] ProcessStartField.process patched: keep Period / recover freq")


def _patch_split_shift_timestamp(default_freq_str: str):
    import gluonts.transform.split as split
    if getattr(split, "_helper_freq_fallback_patched", False):
        return
    _orig = split._shift_timestamp_helper

    def _helper(ts, freq, offset):
        try:
            use_freq = freq if freq is not None else to_offset(default_freq_str)
        except Exception:
            use_freq = to_offset(default_freq_str)
        return _orig(ts, use_freq, offset)

    split._shift_timestamp_helper = _helper
    split._helper_freq_fallback_patched = True
    _print_once(f"[patch] split._shift_timestamp_helper patched: fallback freq={default_freq_str}")


def _patch_add_time_features(freq_hint: str):
    import gluonts.transform.feature as gf
    try:
        from gluonts.time_feature import time_features_from_frequency_str
    except Exception:
        from gluonts.time_feature import time_features_from_frequency_str  # type: ignore

    A = getattr(gf, "AddTimeFeatures", None)
    if A is None or getattr(A, "_robust_map_transform_patched", False):
        return

    orig_map = A.map_transform

    # -------- inner helpers for time features --------
    def _freq_string(self) -> str:
        f = getattr(self, "freq", None)
        if f is None:
            f = freq_hint or "D"
        return str(f).upper()

    def _compute_time_features_fast_B(length: int, f: str) -> np.ndarray:
        """
        Fast path for BusinessDay-like freqs ('B', 'BM', 'BQ', etc):
        avoid pandas.date_range(BusinessDay) which is slow for long ranges.
        We synthesize simple periodic/position features with the same channel count.
        """
        L = int(length)
        if L <= 0:
            return np.ones((1, 1), dtype=np.float32)

        try:
            M = len(time_features_from_frequency_str(f))
        except Exception:
            M = 3
        M = max(1, int(M))

        t = np.arange(L, dtype=np.float32)
        feats = []

        # channel 1: position within business week [0..1]
        if M >= 1:
            feats.append((t % 5.0) / 4.0)
        # channel 2: normalized trend [0..1]
        if M >= 2:
            feats.append(t / max(1.0, (L - 1)))
        # channel 3: bias
        if M >= 3:
            feats.append(np.ones(L, dtype=np.float32))
        # remaining: harmonic (sin/cos) of weekly cycle
        w = 2.0 * np.pi * (t % 5.0) / 5.0
        k = 3
        while k < M:
            feats.append(np.sin(w)); k += 1
            if k < M:
                feats.append(np.cos(w)); k += 1

        return np.vstack([np.asarray(fa, dtype=np.float32)[None, :] for fa in feats])

    @lru_cache(maxsize=8192)
    def _compute_time_features_cached(start_ns: int, length: int, f: str) -> np.ndarray:
        """
        Cached generic path (for non-B* freqs): use pandas.date_range once per key.
        """
        idx = pd.date_range(pd.Timestamp(start_ns), periods=int(length), freq=to_offset(f))
        fns = time_features_from_frequency_str(f)
        rows = [np.asarray(fn(idx), dtype=np.float32).reshape(1, -1) for fn in fns]
        return _vstack_or_ones(rows, length)

    def _compute_time_features_any(self, start_any, length: int, f: str) -> np.ndarray:
        f_up = (f or "D").upper()
        if f_up.startswith("B"):   # BusinessDay family -> fast path
            return _compute_time_features_fast_B(length, f_up)
        # generic cached path
        try:
            ts_start = _to_ts_start(start_any)
            start_ns = int(pd.Timestamp(ts_start).value)
            try:
                off = to_offset(f_up)
                f_resolved = str(off)
            except Exception:
                f_resolved = "D"
            return _compute_time_features_cached(start_ns, int(length), f_resolved)
        except Exception:
            return np.ones((1, int(length)), dtype=np.float32)

    # -------- patched map_transform --------
    def new_map_transform(self, data: dict, is_train: bool) -> dict:
        f = _freq_string(self)

        # For B* freqs, bypass orig_map to avoid slow BusinessDay date_range
        if f.startswith("B"):
            out = dict(data)
            L = _safe_int_len(out.get("target"))
            start_any = out.get("start")
            tf = _compute_time_features_any(self, start_any, L, f)
            out["time_feat"] = tf
            out["time_features"] = tf
            return out

        # Otherwise, try original path first, then enforce robust fallback
        try:
            out = orig_map(self, data, is_train)
            tf = out.get(getattr(self, "output_field", "time_feat"), None)
            if tf is None:
                tf = out.get("time_features", None)
            if tf is not None:
                tf = _ndim2(tf)
                if tf.shape[0] == 0 or tf.shape[1] == 0:
                    L = _safe_int_len(out.get("target", data.get("target")))
                    tf = np.ones((1, L), dtype=np.float32)
                    out["time_feat"] = tf
                    out["time_features"] = tf
            else:
                L = _safe_int_len(out.get("target", data.get("target")))
                start_any = out.get("start", data.get("start"))
                tf = _compute_time_features_any(self, start_any, L, f)
                out["time_feat"] = tf
                out["time_features"] = tf
            return out
        except KeyError:
            _print_once("[patch] AddTimeFeatures patched: robust fallback on KeyError (recompute date_range)")
        except Exception as e:
            _print_once(f"[patch] AddTimeFeatures patched: robust fallback on error: {repr(e)}")

        out = dict(data)
        L = _safe_int_len(out.get("target"))
        start_any = out.get("start")
        tf = _compute_time_features_any(self, start_any, L, f)
        out["time_feat"] = tf
        out["time_features"] = tf
        return out

    A.map_transform = new_map_transform
    A._robust_map_transform_patched = True
    _print_once("[patch] AddTimeFeatures patched: fast B* path, cached generic path, never 0 columns")


def _patch_lstm_dtype_and_size():
    # Align dtype and adapt feature width to expected input_size by zero-pad/trim
    import torch
    import torch.nn as nn

    if getattr(nn.LSTM, "_dtype_size_guard", False):
        return

    _orig = nn.LSTM.forward

    def _safe(self, input, hx=None):
        # dtype align
        w = getattr(self, "weight_ih_l0", None)
        target_dtype = w.dtype if w is not None else input.dtype
        if input.dtype != target_dtype:
            input = input.to(target_dtype)
        if hx is not None:
            if isinstance(hx, tuple) and len(hx) == 2:
                h0, c0 = hx
                if h0 is not None and h0.dtype != target_dtype:
                    h0 = h0.to(target_dtype)
                if c0 is not None and c0.dtype != target_dtype:
                    c0 = c0.to(target_dtype)
                hx = (h0, c0)
            else:
                hx = hx.to(target_dtype)

        # feature-size align
        expected = int(getattr(self, "input_size", input.size(-1)))
        cur = int(input.size(-1))
        if cur != expected:
            # pad with zeros or trim to match expected
            if cur < expected:
                pad = expected - cur
                pad_shape = list(input.shape[:-1]) + [pad]
                zeros = torch.zeros(*pad_shape, dtype=input.dtype, device=input.device)
                input = torch.cat([input, zeros], dim=-1)
            else:
                input = input[..., :expected]

        return _orig(self, input, hx)

    nn.LSTM.forward = _safe
    nn.LSTM._dtype_size_guard = True
    _print_once("[patch] LSTM.forward patched: auto-align dtype and input_size (pad/trim features)")


# ---------- public ----------
def apply_all(freq_hint: str = "D"):
    _patch_lstm_dtype_and_size()
    try:
        import gluonts  # noqa: F401
        pass
    except Exception:
        pass
    _patch_predictor_freqkw(freq_hint)
    _patch_process_start_field_keep_period(freq_hint)
    _patch_split_shift_timestamp(freq_hint)
    _patch_add_time_features(freq_hint)
