# -*- coding: utf-8 -*-
"""
mg_gluonts_patches.py
集中修复 GluonTS 与 pandas 2.x 在 start/freq 上的兼容问题，并兜底特征堆叠的空拼接异常。
该模块可被上层脚本动态 import 后调用 apply_all(freq_hint) 生效。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

def _patch_process_start_field_keep_period(default_freq_str: str):
    try:
        import gluonts.dataset.common as gdc
    except Exception as e:
        print(f"[mgp] WARN: cannot import gluonts.dataset.common ({e})"); return

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
    print("[patch] ProcessStartField.process patched: keep Period / recover freq")

def _patch_split_shift_timestamp(default_freq_str: str):
    try:
        import gluonts.transform.split as split
    except Exception as e:
        print(f"[mgp] WARN: cannot import gluonts.transform.split ({e})"); return

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
    print(f"[patch] split._shift_timestamp_helper patched: fallback freq={default_freq_str}")

def _patch_gluonts_feature_update_cache():
    try:
        import gluonts.transform.feature as gf
    except Exception as e:
        print(f"[mgp] WARN: cannot import gluonts.transform.feature ({e})"); return

    patched = False
    for name, obj in vars(gf).items():
        if isinstance(obj, type) and hasattr(obj, "_update_cache"):
            meth = getattr(obj, "_update_cache")
            if getattr(meth, "_patched_period_safe", False):
                continue

            def new_update_cache(self, start, length, _orig=meth):
                try:
                    freq_str = getattr(self, "freq", None)
                    if isinstance(start, pd.Period):
                        if freq_str is None:
                            try: freq_str = start.freqstr
                            except Exception: pass
                        ts_start = start.to_timestamp(how="start")
                    else:
                        ts_start = pd.Timestamp(start)
                    if freq_str is None:
                        freq_str = "D"
                    try:
                        off = to_offset(freq_str)
                        setattr(self, "_freq_base", getattr(off, "base", None))
                    except Exception:
                        try: setattr(self, "_freq_base", None)
                        except Exception: pass
                    self.full_date_range = pd.date_range(ts_start, periods=length, freq=to_offset(freq_str))
                except Exception:
                    return _orig(self, start, length)

            new_update_cache._patched_period_safe = True
            setattr(obj, "_update_cache", new_update_cache)
            patched = True
    if patched:
        print("[patch] GluonTS feature._update_cache patched for Period/Timestamp")
    else:
        print("[patch] No suitable _update_cache found to patch (maybe already compatible).")

def _patch_gluonts_convert_safe():
    try:
        import gluonts.transform.convert as gconv
    except Exception as e:
        print(f"[mgp] WARN: cannot import gluonts.transform.convert ({e})"); return

    patched = False

    def _length_from_any(data, keys):
        for k in keys + ["target", "feat_dynamic_real", "feat_age"]:
            v = data.get(k, None)
            if v is None:
                continue
            try:
                arr = np.asarray(v)
                if arr.ndim >= 1 and arr.shape[-1] > 0:
                    return int(arr.shape[-1])
            except Exception:
                pass
        return 1

    def _wrap_cls(C):
        nonlocal patched
        if C is None or not hasattr(C, "transform") or getattr(C, "_safe_empty_patch", False):
            return
        orig = C.transform
        def new_transform(self, data):
            try:
                return orig(self, data)
            except ValueError as e:
                if "need at least one array to concatenate" not in str(e):
                    raise
                try:
                    input_fields = list(getattr(self, "input_fields", []))
                except Exception:
                    input_fields = []
                L = _length_from_any(data, input_fields)
                data[getattr(self, "output_field", "feat_dynamic_real")] = np.zeros((1, L), dtype=float)
                return data
        C.transform = new_transform
        C._safe_empty_patch = True
        patched = True

    for name in ("Convert", "VstackFeatures", "HstackFeatures", "HStackFeatures"):
        _wrap_cls(getattr(gconv, name, None))

    if patched:
        print("[patch] GluonTS convert.* patched: safe empty vstack/hstack")
    else:
        print("[patch] WARN: no convert.* class patched (maybe already ok)")

def apply_all(freq_hint: str = "D"):
    """
    一次性应用全部补丁；freq_hint 取自数据集 YAML 的 dataset.freq（如 B/D/H）。
    """
    _patch_process_start_field_keep_period(freq_hint)
    _patch_split_shift_timestamp(freq_hint)
    _patch_gluonts_feature_update_cache()
    _patch_gluonts_convert_safe()
