# -*- coding: utf-8 -*-
"""
mg_gluonts_patches.py
统一打补丁：
- LSTM.forward: 自动对齐 input/hx 到权重 dtype
- PyTorchPredictor.__init__: 兼容 freq/frequency，缺省注入默认频率
- GluonTS ProcessStartField/split/feature：Period/Timestamp 兼容 + _date_index
- AddTimeFeatures：保证不返回 0 列
- mgtsd{Training,Prediction}Network.unroll：健壮处理 time_feat/idx_emb 的长度为 0 或长度不匹配；兼容 *args/**kwargs
"""

from __future__ import annotations
import inspect
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset


def _patch_lstm_dtype():
    import torch.nn as nn
    if getattr(nn.LSTM, "_dtype_align_guard", False):
        return
    _orig = nn.LSTM.forward

    def _safe(self, input, hx=None):
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
        return _orig(self, input, hx)

    nn.LSTM.forward = _safe
    nn.LSTM._dtype_align_guard = True
    print("[patch] LSTM.forward patched: auto-align input/state dtype to weights")


def _patch_predictor_freqkw(default_freq_str: str):
    def _patch_cls(cls, label):
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
            print(f"[patch] estimator.PyTorchPredictor patched: ensure freq/frequency (default={default_freq_str})")
            return True
        except Exception:
            return False

    done = False
    for dotted, label in [
        ("estimator", "estimator.PyTorchPredictor"),
        ("mgtsd_estimator", "mgtsd_estimator.PyTorchPredictor"),
        ("gluonts.torch.model.predictor", "gluonts.torch.model.predictor.PyTorchPredictor"),
    ]:
        try:
            mod = __import__(dotted, fromlist=["*"])
            if hasattr(mod, "PyTorchPredictor"):
                done |= _patch_cls(mod.PyTorchPredictor, label)
        except Exception:
            continue
    if not done:
        print("[patch] no PyTorchPredictor needed patching (already compatible)")


def _patch_process_start_field_keep_period(default_freq_str: str):
    try:
        import gluonts.dataset.common as gdc
    except Exception as e:
        print(f"[patch] WARN: cannot import gluonts.dataset.common ({e})")
        return
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
        print(f"[patch] WARN: cannot import gluonts.transform.split ({e})")
        return
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
        print(f"[patch] WARN: failed to import gluonts.transform.feature ({e})")
        return

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
                            try:
                                freq_str = start.freqstr
                            except Exception:
                                pass
                        ts_start = start.to_timestamp(how="start")
                    else:
                        ts_start = pd.Timestamp(start)
                    if freq_str is None:
                        freq_str = "D"
                    off = to_offset(freq_str)
                    self.full_date_range = pd.date_range(ts_start, periods=int(length), freq=off)
                    try:
                        self._date_index = {ts: i for i, ts in enumerate(self.full_date_range)}
                    except Exception:
                        pass
                except Exception:
                    return _orig(self, start, length)

            new_update_cache._patched_period_safe = True
            setattr(obj, "_update_cache", new_update_cache)
            patched = True
    if patched:
        print("[patch] GluonTS feature._update_cache patched for Period/Timestamp (+_date_index)")
    else:
        print("[patch] No suitable _update_cache found to patch (maybe already compatible).")


def _patch_add_time_features_nonempty(default_freq_str: str):
    # 兜底：AddTimeFeatures 至少返回 1 列；若时间长度为 0，按 target/past_target 长度兜底。
    try:
        import gluonts.transform.feature as gf
    except Exception as e:
        print(f"[patch] WARN: cannot import gluonts.transform.feature for AddTimeFeatures patch ({e})")
        return

    C = getattr(gf, "AddTimeFeatures", None)
    if C is None or not hasattr(C, "map_transform"):
        print("[patch] WARN: AddTimeFeatures not found; skip non-empty guard")
        return
    if getattr(C.map_transform, "_nonempty_guard_v2", False):
        return

    orig = C.map_transform

    def new_map_transform(self, data, is_train):
        out = orig(self, data, is_train)
        key = "time_feat"
        tf = out.get(key, None)

        def _infer_T():
            for k in ("target", "past_target", "feat_age"):
                v = out.get(k, None)
                if v is not None:
                    try:
                        a = np.asarray(v)
                        if a.ndim >= 1:
                            return int(a.shape[-1])
                    except Exception:
                        pass
            return 1

        try:
            arr = np.asarray(tf) if tf is not None else None
            if arr is None or arr.size == 0:
                T = _infer_T()
                arr = np.ones((T, 1), dtype=np.float32)
            else:
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                elif arr.ndim == 2 and arr.shape[1] == 0:
                    arr = np.ones((arr.shape[0], 1), dtype=np.float32)
        except Exception:
            T = _infer_T()
            arr = np.ones((T, 1), dtype=np.float32)

        out[key] = arr
        return out

    new_map_transform._nonempty_guard_v2 = True
    C.map_transform = new_map_transform
    print("[patch] AddTimeFeatures patched: never return 0 columns (fallback to constant ones)")


def _patch_convert_safe():
    try:
        import gluonts.transform.convert as gconv
    except Exception as e:
        print(f"[patch] WARN: failed to import gluonts.transform.convert ({e})")
        return

    patched = False

    def _length_from_any(data, keys):
        for k in list(keys) + ["target", "feat_dynamic_real", "feat_age"]:
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

    print("[patch] GluonTS convert.* patched: safe empty vstack/hstack" if patched
          else "[patch] WARN: no convert.* class patched (maybe already ok)")


def _patch_mgtsd_unroll_safe():
    """
    关键补丁：稳健拼接 input_lags / repeated_index_embeddings / time_feat
    - 同时兼容 unroll(*args) 与 unroll(**kwargs)
    - 若 time_feat 长度为 0 或 None -> 生成 (B, L, 1) 的常量 1
    - 若 repeat_emb/time_feat 的时间长度 != L -> 自动对齐到 L
    """
    try:
        import torch
        import mgtsd_network as net
    except Exception as e:
        print(f"[patch] WARN: cannot import mgtsd_network for unroll patch ({e})")
        return

    def _wrap_unroll(cls, label):
        if not hasattr(cls, "unroll"):
            return False
        orig = cls.unroll
        if getattr(orig, "_safe_cat_patched_v2", False):
            return False

        def new_unroll(self, *args, **kwargs):
            def _safe_get(d, primary, *alts):
                if primary in d:
                    return d[primary]
                for k in alts:
                    if k in d:
                        return d[k]
                return None

            def _extract(a, kw):
                # 位置参数形式
                if len(a) >= 5:
                    rnn, begin_state, input_lags, repeated_index_embeddings, time_feat = a[:5]
                    tail = a[5:]
                    return (rnn, begin_state, input_lags, repeated_index_embeddings, time_feat, tail, {})
                # 关键字参数
                keys = kw.keys()
                need = ("rnn", "begin_state", "input_lags", "repeated_index_embeddings", "time_feat")
                if all(k in keys for k in need):
                    rnn = kw["rnn"]; begin_state = kw["begin_state"]
                    input_lags = kw["input_lags"]
                    repeated_index_embeddings = kw["repeated_index_embeddings"]
                    time_feat = kw["time_feat"]
                    other = {k: v for k, v in kw.items() if k not in need}
                    return (rnn, begin_state, input_lags, repeated_index_embeddings, time_feat, (), other)
                # 常见别名（不能用 “or” 以免触发张量布尔求值）
                input_lags = _safe_get(kw, "input_lags", "lags")
                repeated_index_embeddings = _safe_get(kw, "repeated_index_embeddings", "idx_emb")
                time_feat = _safe_get(kw, "time_feat", "time_features")
                rnn = kw.get("rnn", None)
                begin_state = kw.get("begin_state", None)
                other = {k: v for k, v in kw.items()
                         if k not in ("input_lags", "lags",
                                      "repeated_index_embeddings", "idx_emb",
                                      "time_feat", "time_features", "rnn", "begin_state")}
                return (rnn, begin_state, input_lags, repeated_index_embeddings, time_feat, (), other)

            # 先尝试原始调用（大部分情况直接成功）
            try:
                return orig(self, *args, **kwargs)
            except RuntimeError as e:
                msg = str(e)
                needs_fix = ("Sizes of tensors must match" in msg) or ("for tensor number" in msg)
                if not needs_fix:
                    raise
            except ValueError:
                # e.g. not enough values to unpack
                pass

            # 进入修复路径
            rnn, begin_state, input_lags, idx_emb, time_feat, tail, other = _extract(args, kwargs)

            if input_lags is None:
                return orig(self, *args, **kwargs)

            B = int(input_lags.size(0))
            L = int(input_lags.size(1))

            # 处理 idx_emb
            if idx_emb is None:
                idx_emb = torch.zeros(B, L, 0, dtype=input_lags.dtype, device=input_lags.device)
            else:
                if idx_emb.dim() == 2:
                    # (B, C) -> (B, L, C)
                    idx_emb = idx_emb.unsqueeze(1).expand(B, L, idx_emb.size(-1))
                elif idx_emb.dim() == 3 and idx_emb.size(1) != L:
                    if idx_emb.size(1) == 1:
                        idx_emb = idx_emb.expand(B, L, idx_emb.size(2))
                    elif idx_emb.size(1) < L:
                        pad_len = L - idx_emb.size(1)
                        pad = torch.zeros(B, pad_len, idx_emb.size(2), dtype=idx_emb.dtype, device=idx_emb.device)
                        idx_emb = torch.cat([idx_emb, pad], dim=1)
                    else:
                        idx_emb = idx_emb[:, :L, :]

            # 处理 time_feat
            if time_feat is None or time_feat.numel() == 0:
                time_feat = torch.ones(B, L, 1, dtype=input_lags.dtype, device=input_lags.device)
            else:
                # 统一为 (B, L, K)
                if time_feat.dim() == 2:
                    # 可能是 (L, K) 或 (B, K)；优先按 (L, K) 解释
                    if time_feat.size(0) == L:
                        time_feat = time_feat.unsqueeze(0).expand(B, L, time_feat.size(1))
                    else:
                        time_feat = time_feat.unsqueeze(1).expand(B, L, time_feat.size(1))
                elif time_feat.dim() == 1:
                    time_feat = time_feat.view(1, 1, -1).expand(B, L, -1)
                elif time_feat.dim() == 3 and time_feat.size(1) != L:
                    if time_feat.size(1) == 1:
                        time_feat = time_feat.expand(B, L, time_feat.size(2))
                    elif time_feat.size(1) < L:
                        pad_len = L - time_feat.size(1)
                        pad = torch.ones(B, pad_len, time_feat.size(2), dtype=time_feat.dtype, device=time_feat.device)
                        time_feat = torch.cat([time_feat, pad], dim=1)
                    else:
                        time_feat = time_feat[:, :L, :]

            # 重新组织参数，再调一次原函数
            if len(args) >= 5:
                new_args = (rnn, begin_state, input_lags, idx_emb, time_feat) + tail
                return orig(self, *new_args, **other)
            else:
                new_kwargs = dict(other)
                if rnn is not None: new_kwargs["rnn"] = rnn
                if begin_state is not None: new_kwargs["begin_state"] = begin_state
                new_kwargs["input_lags"] = input_lags
                new_kwargs["repeated_index_embeddings"] = idx_emb
                new_kwargs["time_feat"] = time_feat
                return orig(self, **new_kwargs)

        new_unroll._safe_cat_patched_v2 = True
        cls.unroll = new_unroll
        print(f"[patch] {label}.unroll patched: safe-cat with zero-length/mismatch handling")
        return True

    any_done = False
    try:
        import mgtsd_network as net
        any_done |= _wrap_unroll(getattr(net, "mgtsdPredictionNetwork", None), "mgtsdPredictionNetwork")
        any_done |= _wrap_unroll(getattr(net, "mgtsdTrainingNetwork", None), "mgtsdTrainingNetwork")
    except Exception as e:
        print(f"[patch] WARN: mgtsd_*Network import failed ({e})")
        any_done = False

    if not any_done:
        print("[patch] WARN: mgtsd_*Network.unroll not patched (class not found?)")


def apply_all(freq_hint: str = "B"):
    # 先打 PyTorch / 低层补丁
    _patch_lstm_dtype()

    # GluonTS / Predictor 相关
    _patch_predictor_freqkw(freq_hint)
    print("self.log_metrics: None")

    _patch_process_start_field_keep_period(freq_hint)
    _patch_split_shift_timestamp(freq_hint)
    _patch_gluonts_feature_update_cache()
    _patch_add_time_features_nonempty(freq_hint)
    _patch_convert_safe()

    # MG-TSD 网络层关键补丁
    _patch_mgtsd_unroll_safe()
