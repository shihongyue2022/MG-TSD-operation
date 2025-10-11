# /home/shihongyue/MG-TSD-operation/src/mgtsd_adapter.py
from __future__ import annotations
import importlib
import inspect
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional, Iterable, Tuple, List

import numpy as np
import torch
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.exceptions import GluonTSDataError

# ========== PATCH: 强化 GluonTS 时间特征构造，避免 _date_index 断言/索引错误 ==========
try:
    from gluonts.transform.feature import AddTimeFeatures as _ATF  # type: ignore
    _orig_update_cache = getattr(_ATF, "_update_cache", None)
    _orig_map_transform = getattr(_ATF, "map_transform", None)

    def _to_rule_code(x) -> Optional[str]:
        try:
            # pandas offset -> 规则码，例如 <BusinessDay> -> "B"
            return getattr(x, "rule_code", None) or getattr(x, "freqstr", None) or str(x)
        except Exception:
            return None

    def _norm_start_and_freq(start, fallback_freq: str = "B") -> Tuple[pd.Timestamp, str]:
        """把 start 归一到 Timestamp，并返回稳定的 freq 字符串。"""
        if isinstance(start, pd.Period):
            freq_str = start.freqstr or fallback_freq
            return start.start_time, str(freq_str)
        if isinstance(start, pd.Timestamp):
            # 有些版本 Timestamp.freq 为空；优先用 fallback
            fs = getattr(start, "freq", None)
            freq_str = _to_rule_code(fs) or fallback_freq
            return start, str(freq_str)
        try:
            p = pd.Period(str(start), freq=fallback_freq)
            return p.start_time, fallback_freq
        except Exception:
            return pd.Timestamp("2000-01-01"), fallback_freq

    if callable(_orig_update_cache):
        def _patched_update_cache(self, start, length):
            # 统一为 Timestamp + 明确的 freq 字符串
            freq_attr = getattr(self, "freq", None)
            fb = _to_rule_code(freq_attr) or "B"
            start_ts, freq_str = _norm_start_and_freq(start, fb)

            # 优先走原实现
            try:
                return _orig_update_cache(self, start_ts, length)
            except Exception:
                # 兜底：构造 full_date_range，并将 _date_index 设为 dict{Timestamp: idx}
                try:
                    self.full_date_range = pd.date_range(start=start_ts, periods=int(length), freq=freq_str)
                except Exception:
                    self.full_date_range = pd.date_range(start="2000-01-01", periods=int(length), freq="B")
                try:
                    self._date_index = {ts: i for i, ts in enumerate(self.full_date_range)}
                except Exception:
                    # 再兜底：转 Timestamp 强制一致
                    self._date_index = {pd.Timestamp(ts): i for i, ts in enumerate(list(self.full_date_range))}
                return None
        _ATF._update_cache = _patched_update_cache  # type: ignore

    if callable(_orig_map_transform):
        def _patched_map_transform(self, data, is_train):
            # 统一 start -> Timestamp；并提前确保 _date_index 是 dict
            freq_attr = getattr(self, "freq", None)
            fb = _to_rule_code(freq_attr) or "B"

            start_orig = data.get(getattr(self, "start_field", "start"))
            start_ts, _ = _norm_start_and_freq(start_orig, fb)
            data[getattr(self, "start_field", "start")] = start_ts  # 关键：传 Timestamp 而不是 Period

            # 估计序列长度（target 必有）
            tgt = data.get(getattr(self, "target_field", "target"))
            try:
                length = int(tgt.shape[-1]) if hasattr(tgt, "shape") else int(len(tgt))
            except Exception:
                length = 1024

            # 先手动更新一次 cache，避免 _date_index 为 None
            try:
                self._update_cache(start_ts, length)
            except Exception:
                pass

            # 确保 _date_index 为 dict（原库需要 _date_index[start] 返回整型位置）
            if not isinstance(getattr(self, "_date_index", None), dict):
                try:
                    fr = getattr(self, "full_date_range", None)
                    if fr is None:
                        fr = pd.date_range(start=start_ts, periods=length, freq=fb)
                        self.full_date_range = fr
                    self._date_index = {ts: i for i, ts in enumerate(fr)}
                except Exception:
                    dr = pd.date_range(start="2000-01-01", periods=length, freq="B")
                    self.full_date_range = dr
                    self._date_index = {ts: i for i, ts in enumerate(dr)}

            # 再交回原实现（此时 _date_index[start_ts] 一定是整数）
            try:
                return _orig_map_transform(self, data, is_train)
            except AssertionError:
                # 极端兜底：若仍断言，则重建 _date_index 后再试一次
                fr = getattr(self, "full_date_range", pd.date_range(start=start_ts, periods=length, freq=fb))
                self._date_index = {ts: i for i, ts in enumerate(fr)}
                return _orig_map_transform(self, data, is_train)

        _ATF.map_transform = _patched_map_transform  # type: ignore

    print("[patch] GluonTS AddTimeFeatures patched: Timestamp start + dict(_date_index)")
except Exception as _e:
    print(f"[patch] WARN: failed to patch AddTimeFeatures ({_e})")
# =====================================================================


# ---------- deps stubs & compat aliases ----------

def _ensure_wandb_stub():
    if "wandb" in sys.modules:
        return
    m = types.ModuleType("wandb")
    def _noop(*args, **kwargs): pass
    m.init = _noop; m.login = _noop; m.finish = _noop
    m.watch = _noop; m.unwatch = _noop; m.log = _noop
    m.config = {}; m.run = None
    sys.modules["wandb"] = m

def _install_gluonts_compat_aliases():
    # 兼容部分旧版 gluonts 的包路径差异
    try:
        importlib.import_module("gluonts.torch.modules.distribution_output")
    except ModuleNotFoundError:
        try:
            dist_mod = importlib.import_module("gluonts.torch.distributions")
        except Exception:
            return
        if "gluonts.torch.modules" not in sys.modules:
            pkg = types.ModuleType("gluonts.torch.modules")
            pkg.__path__ = []
            sys.modules["gluonts.torch.modules"] = pkg
        sys.modules["gluonts.torch.modules.distribution_output"] = dist_mod

def _safe_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name == "wandb":
            _ensure_wandb_stub()
            return importlib.import_module(module_name)
        raise

def _sig_accepts(obj, name: str) -> bool:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if name in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

def _filter_kwargs(obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    # 仅保留 obj 可接受的关键字（或带 **kwargs 的）
    try:
        sig = inspect.signature(obj)
    except Exception:
        return {}
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}

def _resolve_estimator_cls():
    _install_gluonts_compat_aliases()
    last_err = None
    try:
        m = _safe_import("mgtsd_estimator")
    except Exception as e:
        last_err = e
        m = None
    if m is None:
        raise ImportError(f"无法导入 mgtsd_estimator（最后错误：{last_err})")

    # 选择实现了自定义 train 的类
    try:
        import gluonts.model.estimator as ge  # type: ignore
        base_train = ge.Estimator.train
    except Exception:
        base_train = None

    candidates: List[Tuple[str, type]] = []
    for name, obj in vars(m).items():
        if isinstance(obj, type) and hasattr(obj, "train"):
            if getattr(obj, "__module__", "") != m.__name__:
                continue
            if base_train is not None and getattr(obj, "train", None) is base_train:
                continue
            candidates.append((name, obj))
    if not candidates:
        raise ImportError("找不到可用的 Estimator 类：请确认 mgtsd_estimator.py 中存在自定义类，并实现了 train(...)。")

    def _rank(name: str) -> Tuple[int, int]:
        nm = name.lower()
        return (0 if ("mgtsd" in nm or "mg" in nm) else 1, len(nm))
    candidates.sort(key=lambda kv: _rank(kv[0]))
    chosen_name, chosen_cls = candidates[0]
    print(f"[mgtsd_adapter] 使用 Estimator 类: {m.__name__}.{chosen_name}")
    return chosen_cls

# ---------- 小工具 ----------

def _parse_list(s: Any, tp=float, sep=r"[,_\s]+") -> List:
    if s is None:
        return []
    if isinstance(s, (list, tuple, np.ndarray)):
        return [tp(x) for x in s]
    parts = re.split(sep, str(s).strip())
    return [tp(p) for p in parts if p != ""]

def _as_1d_target(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim >= 2:
        arr = arr.reshape(-1)
    return arr

def _as_2d_target(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim >= 2 and arr.shape[0] != 1:
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        else:
            arr = arr[:1, :]
    return arr

def _to_period(start_like: Any, freq: str) -> pd.Period:
    if isinstance(start_like, pd.Period):
        return pd.Period(start_like.start_time, freq=freq)
    if isinstance(start_like, pd.Timestamp):
        return start_like.to_period(freq)
    try:
        return pd.Period(str(start_like), freq=freq)
    except Exception:
        return pd.Period("2000-01-01", freq=freq)

def _need_2d(msg: str) -> bool:
    s = msg.lower()
    return ("field" in s and "target" in s and "observed: 1" in s and "expected ndim: 2" in s) or \
           ("expected" in s and "ndim" in s and "2" in s)

def _need_1d(msg: str) -> bool:
    s = msg.lower()
    return ("field" in s and "target" in s and "observed: 2" in s and "expected ndim: 1" in s) or \
           ("expected" in s and "ndim" in s and "1" in s)

def _parse_input_size_mismatch(msg: str) -> Optional[int]:
    m = re.search(r"Expected\s+(\d+),\s*got\s+(\d+)", msg)
    if m:
        exp, got = int(m.group(1)), int(m.group(2))
        return max(got, 1)
    return None

def _is_lag_hist_assert(msg: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"found lag\s+(\d+)\s+while history length is only\s+(\d+)", msg)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

# ---------- 无校验数据集包装 ----------

class _RawListDataset:
    def __init__(self, entries: Iterable[Dict[str, Any]], freq: str):
        self._data = list(entries)
        self.freq = freq
    def __iter__(self):
        for x in self._data:
            yield x
    def __len__(self):
        return len(self._data)
    def __getitem__(self, idx):
        return self._data[idx]

# ---------- Adapter ----------

class MGAdapter:
    """
    只通过“适配器/数据侧”兜底，不改动模型：
    - 1D/2D target 自动切换
    - 自动左侧复制填充(extend-left) 让序列长度满足 max_lag + context + pred (+安全边际)
    - 捕获 input_size 不匹配 -> 重建 Estimator(input_size=实际特征维)
    - 捕获 lags 超历史长度 -> 将 lags 压到 [1] + 调整 context
    - 捕获 circular padding 报错 -> 仅通过加长上下文/放大训练用 pred_len（不动模型）
    """
    def __init__(self, cfg: Dict[str, Any], device: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.predictor = None
        self._trained_expect_2d = False

        # 超参（来自 cfg 顶层或 mgtsd 子块）
        self.epoch       = int(cfg.get("epoch", cfg.get("epochs", 30)))
        self.diff_steps  = int(cfg.get("diff_steps", 100))
        self.batch_size  = int(cfg.get("batch_size", 128))
        self.num_cells   = int(cfg.get("num_cells", 64))
        self.mg_dict_raw = cfg.get("mg_dict", "1_4_12")
        self.num_gran    = int(cfg.get("num_gran", 3))
        self.share_ratio_raw = cfg.get("share_ratio_list", "1_0.8_0.6")
        self.weight_list_raw = cfg.get("weight_list", "0.8_0.1_0.1")
        self.ckpt_path   = Path(cfg.get("ckpt_path", "")).expanduser()

        # 关键维度
        self.input_size  = cfg.get("input_size")
        self.target_dim  = cfg.get("target_dim")

        # 解析 list
        self.mg_list     = _parse_list(self.mg_dict_raw, tp=int)
        self.share_list  = _parse_list(self.share_ratio_raw, tp=float)
        self.weight_list = _parse_list(self.weight_list_raw, tp=float)
        if self.weight_list and self.mg_list and len(self.weight_list) != len(self.mg_list):
            if len(self.weight_list) < len(self.mg_list):
                last = self.weight_list[-1]
                self.weight_list = self.weight_list + [last] * (len(self.mg_list) - len(self.weight_list))
            else:
                self.weight_list = self.weight_list[:len(self.mg_list)]

        # 适配层控制
        self.min_ctx_floor = int(cfg.get("min_ctx_floor", 64))
        self.max_lag_guess = int(cfg.get("max_lag_guess", 7))
        self.safety_margin = int(cfg.get("safety_margin", 24))

        # circular padding 兜底策略
        self.max_pad_bumps_ctx = int(cfg.get("max_pad_bumps_ctx", 8))
        self.max_pred_bumps    = int(cfg.get("max_pred_bumps", 4))
        self.pred_bump_growth  = float(cfg.get("pred_bump_growth", 1.5))
        self.pred_len_cap      = int(cfg.get("pred_len_cap", 512))
        self.max_retries       = int(cfg.get("max_retries", 60))

    @staticmethod
    def _infer_dims_from_items(items: Iterable[Dict[str, Any]]) -> Tuple[int, int]:
        first = None
        for it in items:
            first = it
            break
        if first is None:
            return 1, 1
        arr = np.asarray(first["target"], dtype=float)
        if arr.ndim == 1:
            return 1, 1
        elif arr.ndim == 2:
            dim = int(arr.shape[0])
            return dim, dim
        return 1, 1

    # -------- Estimator 构建：稳健处理 num_gran --------
    def _build_estimator(
        self,
        prediction_length:int,
        context_length:int,
        freq:str,
        num_parallel_samples:int,
        force_input_size: Optional[int] = None
    ):
        Estimator = _resolve_estimator_cls()

        # 单变量时把 mg 组等裁剪到 1 组，避免 split 报错
        if (self.input_size or 1) == 1 and self.num_gran > 1:
            print(f"[mgtsd_adapter] 单变量输入(input_size=1) -> 将 num_gran {self.num_gran} 降为 1，并裁剪 mg/share/weight 列表")
            self.num_gran = 1
            if len(self.mg_list) > 1:     self.mg_list = self.mg_list[:1]
            if len(self.share_list) > 1:  self.share_list = self.share_list[:1]
            if len(self.weight_list) > 1: self.weight_list = self.weight_list[:1]

        core_kwargs = dict(
            prediction_length=int(prediction_length),
            context_length=int(context_length),
            freq=freq,
            input_size=int(force_input_size if force_input_size is not None else (self.input_size or 1)),
            target_dim=int(self.target_dim or 1),
            num_gran=int(self.num_gran),
        )

        try:
            est = Estimator(**core_kwargs)
        except TypeError as e1:
            msg = str(e1)
            if "unexpected keyword argument 'num_gran'" in msg or "got an unexpected keyword argument 'num_gran'" in msg:
                kw2 = dict(core_kwargs); kw2.pop("num_gran", None)
                try:
                    est = Estimator(**kw2)
                except TypeError as e2:
                    if "missing 1 required positional argument: 'num_gran'" in str(e2):
                        est = Estimator(int(self.num_gran), **kw2)
                    else:
                        raise
            elif "missing 1 required positional argument: 'num_gran'" in msg:
                kw2 = dict(core_kwargs); kw2.pop("num_gran", None)
                est = Estimator(int(self.num_gran), **kw2)
            else:
                raise

        _try_inject_hparams(
            est,
            mg_list=self.mg_list,
            share_list=self.share_list,
            weight_list=self.weight_list,
            target_dim=int(self.target_dim or 1),
            diff_steps=self.diff_steps,
            num_cells=self.num_cells,
            device=str(self.device),
            batch_size=self.batch_size,
            epochs=self.epoch,
        )
        return est

    # ---------- 数据集构造（仅数据侧兜底） ----------

    def _min_required_len(self, ctx:int, pred:int, max_lag:int) -> int:
        ctx_eff = max(ctx, self.min_ctx_floor)
        return max_lag + ctx_eff + pred + self.safety_margin

    def _extend_left(self, arr: np.ndarray, need_len: int) -> np.ndarray:
        L = arr.shape[-1]
        if L >= need_len:
            return arr
        pad = need_len - L
        if arr.ndim == 1:
            pad_val = float(arr[0]) if L > 0 else 0.0
            left = np.full((pad,), pad_val, dtype=float)
            return np.concatenate([left, arr.astype(float)], axis=0)
        else:
            pad_val = arr[:, 0:1] if L > 0 else np.zeros((arr.shape[0],1), dtype=float)
            left = np.repeat(pad_val, pad, axis=1)
            return np.concatenate([left, arr.astype(float)], axis=1)

    def _normalize_items(
        self,
        raw_items: Iterable[Dict[str, Any]],
        freq: str,
        expect_2d: bool,
        ctx: int,
        pred: int,
        max_lag: int
    ) -> List[Dict[str, Any]]:
        need_len = self._min_required_len(ctx, pred, max_lag)
        out = []
        for it in raw_items:
            tgt = it["target"]
            arr = _as_2d_target(tgt) if expect_2d else _as_1d_target(tgt)
            arr = self._extend_left(arr, need_len)
            out.append({"target": arr, "start": _to_period(it.get("start", "2000-01-01"), freq)})
        return out

    # ---------- 训练主流程（只改适配器） ----------

    def _train_once(self, estimator, train_ds):
        train_kwargs = _filter_kwargs(estimator.train, dict(training_data=train_ds))
        return estimator.train(**train_kwargs)

    def _train_with_retries(
        self,
        estimator,
        items: List[Dict[str, Any]],
        freq: str,
        prediction_length: int,
        context_length: int,
        expect_2d: bool,
        max_lag: int,
    ) -> Tuple[Any, bool, int]:
        tries = 0
        cur_expect_2d = expect_2d
        cur_max_lag = max_lag
        cur_min_ctx = self.min_ctx_floor
        cur_force_input_size: Optional[int] = None
        cur_ctx = int(context_length)
        cur_train_pred = int(prediction_length)

        bumps_ctx = 0
        bumps_pred = 0

        while True:
            tries += 1
            if tries > self.max_retries:
                raise RuntimeError(f"[mgtsd_adapter] 超过最大重试次数({self.max_retries})，请检查配置/模型。")

            norm = self._normalize_items(items, freq, cur_expect_2d, cur_ctx, cur_train_pred, cur_max_lag)
            ds = (_RawListDataset if cur_expect_2d else ListDataset)(norm, freq=freq)

            try:
                est = self._build_estimator(
                    prediction_length=cur_train_pred,
                    context_length=cur_ctx,
                    freq=freq,
                    num_parallel_samples=cur_train_pred,
                    force_input_size=cur_force_input_size,
                )
                for k in ("lags_seq", "lags", "lag_indices", "lag_seq"):
                    if hasattr(est, k):
                        setattr(est, k, [1] if cur_max_lag <= 1 else list(range(1, cur_max_lag + 1)))

                pred = self._train_once(est, ds)
                self._trained_expect_2d = cur_expect_2d
                return pred, cur_expect_2d, cur_max_lag

            except GluonTSDataError as ge:
                msg = str(ge)
                if _need_2d(msg) and not cur_expect_2d:
                    cur_expect_2d = True
                    continue
                if _need_1d(msg) and cur_expect_2d:
                    cur_expect_2d = False
                    continue
                raise

            except AssertionError as ae:
                text = str(ae)
                lag_hist = _is_lag_hist_assert(text)
                if lag_hist is not None:
                    found_lag, hist = lag_hist
                    allowed_ctx = hist - cur_train_pred - max(found_lag, cur_max_lag)
                    new_ctx = max(self.min_ctx_floor, min(cur_ctx, allowed_ctx))
                    if new_ctx < cur_ctx:
                        print(f"[mgtsd_adapter] 解析到 history={hist}, max_lag={max(found_lag, cur_max_lag)} -> 自动下调 context_length: {cur_ctx} -> {new_ctx}")
                        cur_ctx = new_ctx
                    if cur_max_lag != 1:
                        print(f"[mgtsd_adapter] 将 lags 收敛到 [1]（原最大滞后 {cur_max_lag}）")
                    cur_max_lag = 1
                    continue
                raise

            except RuntimeError as re_err:
                text = str(re_err)
                got = _parse_input_size_mismatch(text)
                if got is not None and (cur_force_input_size or (self.input_size or 1)) != got:
                    print(f"[mgtsd_adapter] 侦测到 input_size 不匹配：Estimator.input_size="
                          f"{(cur_force_input_size or self.input_size or 1)}, 实际特征维={got} -> 重建 Estimator(input_size={got}) 并重训")
                    cur_force_input_size = got
                    continue

                if "Padding value causes wrapping around more than once" in text:
                    if bumps_ctx < self.max_pad_bumps_ctx:
                        new_floor = 128 if cur_min_ctx < 128 else cur_min_ctx + 96
                        print(f"[mgtsd_adapter] 侦测到 circular padding -> 提高最小上下文下限: {cur_min_ctx} -> {new_floor}")
                        cur_min_ctx = new_floor
                        self.min_ctx_floor = cur_min_ctx
                        cur_ctx = max(cur_ctx, cur_min_ctx)
                        bumps_ctx += 1
                        continue
                    if bumps_pred < self.max_pred_bumps and cur_train_pred < self.pred_len_cap:
                        new_pred = min(int(max(cur_train_pred * self.pred_bump_growth, cur_train_pred + 1)), self.pred_len_cap)
                        print(f"[mgtsd_adapter] circular padding 仍存在 -> 放大训练用 prediction_length: {cur_train_pred} -> {new_pred} (推理仍按 {prediction_length} 截取)")
                        cur_train_pred = new_pred
                        bumps_pred += 1
                        continue
                    raise RuntimeError(
                        f"[mgtsd_adapter] circular padding 仍无法满足：已抬 context {bumps_ctx} 次、放大训练用 pred_len {bumps_pred} 次(当前 {cur_train_pred})。"
                        f"请检查卷积核/扩张设定，或在外部 config 提高 prediction_length/context_length。"
                    )

                raise

    def fit_or_load(self,
                    train_items: Iterable[Dict[str, Any]],
                    freq: str,
                    prediction_length: int,
                    context_length: int,
                    num_samples_for_predictor: int):
        _ensure_wandb_stub()

        items = list(train_items)

        if self.input_size is None or self.target_dim is None:
            i_size, t_dim = self._infer_dims_from_items(items)
            if self.input_size is None: self.input_size = i_size
            if self.target_dim is None: self.target_dim = t_dim
        self.input_size = int(self.input_size or 1)
        self.target_dim = int(self.target_dim or 1)

        if self.input_size == 1 and self.num_gran > 1:
            print(f"[mgtsd_adapter] 单变量输入(input_size=1) -> 将 num_gran {self.num_gran} 降为 1，并裁剪 mg/share/weight 列表")
            self.num_gran = 1
            if len(self.mg_list) > 1:     self.mg_list = self.mg_list[:1]
            if len(self.share_list) > 1:  self.share_list = self.share_list[:1]
            if len(self.weight_list) > 1: self.weight_list = self.weight_list[:1]

        estimator = self._build_estimator(prediction_length, context_length, freq, num_samples_for_predictor)
        try:
            pred, used_2d, used_max_lag = self._train_with_retries(
                estimator, items, freq, prediction_length, context_length,
                expect_2d=False, max_lag=self.max_lag_guess
            )
            self.predictor = pred
            self._trained_expect_2d = used_2d
            return self
        except Exception:
            estimator = self._build_estimator(prediction_length, context_length, freq, num_samples_for_predictor)
            pred, used_2d, used_max_lag = self._train_with_retries(
                estimator, items, freq, prediction_length, context_length,
                expect_2d=True, max_lag=self.max_lag_guess
            )
            self.predictor = pred
            self._trained_expect_2d = used_2d
            return self

    # ---------- 推理 ----------

    @torch.no_grad()
    def predict_one(self, x_ctx: np.ndarray, pred_len: int, num_samples: int, freq: str = "B") -> np.ndarray:
        assert self.predictor is not None, "predictor 未初始化，请先调用 fit_or_load(...)"

        def _predict_with_ds(ds):
            it = self.predictor.predict(ds)
            fc = next(iter(it))
            S = np.asarray(fc.samples)
            if S.ndim == 3 and S.shape[-1] == 1:
                S = S[..., 0]
            elif S.ndim != 2:
                if S.ndim == 3 and 1 in S.shape:
                    S = np.squeeze(S)
                if S.ndim != 2:
                    raise RuntimeError(f"未知的 samples 维度: {S.shape}")
            if S.shape[1] != pred_len:
                S = S[:, :pred_len]
            if S.shape[0] != num_samples:
                if S.shape[0] > num_samples:
                    S = S[:num_samples, :]
                else:
                    rep = int(np.ceil(num_samples / max(1, S.shape[0])))
                    S = np.concatenate([S] * rep, axis=0)[:num_samples, :]
            return S

        min_need = self._min_required_len(ctx=pred_len, pred=pred_len, max_lag=1)
        if self._trained_expect_2d:
            tgt = _as_2d_target(x_ctx.astype(float))
            tgt = self._extend_left(tgt, min_need)
            ds = _RawListDataset([{"target": tgt, "start": _to_period("2000-01-01", freq)}], freq=freq)
            try:
                return _predict_with_ds(ds)
            except GluonTSDataError:
                tgt1 = _as_1d_target(x_ctx.astype(float))
                tgt1 = self._extend_left(tgt1, min_need)
                ds1 = ListDataset([{"target": tgt1, "start": _to_period("2000-01-01", freq)}], freq=freq)
                return _predict_with_ds(ds1)
        else:
            tgt1 = _as_1d_target(x_ctx.astype(float))
            tgt1 = self._extend_left(tgt1, min_need)
            ds1 = ListDataset([{"target": tgt1, "start": _to_period("2000-01-01", freq)}], freq=freq)
            try:
                return _predict_with_ds(ds1)
            except GluonTSDataError:
                tgt = _as_2d_target(x_ctx.astype(float))
                tgt = self._extend_left(tgt, min_need)
                ds = _RawListDataset([{"target": tgt, "start": _to_period("2000-01-01", freq)}], freq=freq)
                return _predict_with_ds(ds)

# ---------- 注入超参（仅通过属性/方法，不改模型文件） ----------

def _try_inject_hparams(estimator,
                        mg_list: List[int],
                        share_list: List[float],
                        weight_list: List[float],
                        target_dim: int,
                        diff_steps: Optional[int] = None,
                        num_cells: Optional[int] = None,
                        device: Optional[str] = None,
                        batch_size: Optional[int] = None,
                        epochs: Optional[int] = None):
    for name in ("set_hparams", "set_params", "configure"):
        if hasattr(estimator, name) and callable(getattr(estimator, name)):
            try:
                getattr(estimator, name)({
                    "mg_dict": list(mg_list),
                    "share_ratio_list": list(share_list),
                    "weight_list": list(weight_list),
                    "target_dim": int(target_dim),
                    "diff_steps": diff_steps,
                    "num_cells": num_cells,
                    "device": device,
                    "batch_size": batch_size,
                    "epochs": epochs,
                })
                return
            except Exception:
                pass

    def _safe_set(obj, key, val):
        if hasattr(obj, key) and val is not None:
            try: setattr(obj, key, val)
            except Exception: pass

    for k in ("mg_dict", "self_mg_dict", "mg_scales", "scales"):
        _safe_set(estimator, k, list(mg_list))
    for k in ("share_ratio_list", "self_share_ratio", "share_ratios"):
        _safe_set(estimator, k, list(share_list))
    for k in ("weight_list", "self_weight_list", "weights"):
        _safe_set(estimator, k, list(weight_list))
    _safe_set(estimator, "target_dim", int(target_dim))
    _safe_set(estimator, "input_size", 1 if getattr(estimator, "input_size", None) in (None, 0) else getattr(estimator, "input_size"))
    _safe_set(estimator, "diff_steps", diff_steps)
    _safe_set(estimator, "num_cells", num_cells)
    _safe_set(estimator, "device", device)
    _safe_set(estimator, "batch_size", batch_size)
    _safe_set(estimator, "epochs", epochs)
