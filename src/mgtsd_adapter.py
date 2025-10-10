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
    return {k: v for k, v in kwargs.items() if _sig_accepts(obj, k)}

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
    return ("expected" in s and "ndim" in s and "2" in s) or ("required" in s and "dimension" in s and "2" in s)

def _need_1d(msg: str) -> bool:
    s = msg.lower()
    return ("expected" in s and "ndim" in s and "1" in s)

def _parse_input_size_mismatch(msg: str) -> Optional[int]:
    # 例如: "input.size(-1) must be equal to input_size. Expected 1, got 7"
    m = re.search(r"Expected\s+(\d+),\s*got\s+(\d+)", msg)
    if m:
        exp, got = int(m.group(1)), int(m.group(2))
        return max(got, 1)
    return None

def _is_lag_hist_assert(msg: str) -> Optional[Tuple[int, int]]:
    # 例如: "found lag 7 while history length is only 192"
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
    - 自动左侧复制填充，让序列长度满足 max_lag + context + pred + safety
    - 捕获 input_size 不匹配 -> 重建 Estimator(input_size=实际特征维)
    - 捕获 lags 超历史长度 -> 将 lags 压到 [1] 并调整 context
    - 捕获 circular padding 报错 -> **优先**快速放大“训练用 prediction_length”，
      不改推理输出步数
    """

    def __init__(self, cfg: Dict[str, Any], device: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.predictor = None
        self._trained_expect_2d = False

        # 基本超参（从 cfg 里拿，缺省有默认）
        self.epoch       = int(cfg.get("epoch", 30))
        self.diff_steps  = int(cfg.get("diff_steps", 100))
        self.batch_size  = int(cfg.get("batch_size", 128))
        self.num_cells   = int(cfg.get("num_cells", 64))

        # 多粒度相关（单变量会裁剪）
        self.mg_dict_raw = cfg.get("mg_dict", "1_4_12")
        self.num_gran    = int(cfg.get("num_gran", 3))
        self.share_ratio_raw = cfg.get("share_ratio_list", "1_0.8_0.6")
        self.weight_list_raw = cfg.get("weight_list", "0.8_0.1_0.1")

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

        # 适配层参数（可在 YAML 里覆盖）
        self.min_ctx_floor      = int(cfg.get("min_ctx_floor", 64))
        self.max_lag_guess      = int(cfg.get("max_lag_guess", 7))
        self.safety_margin      = int(cfg.get("safety_margin", 24))

        # —— 新增：针对 circular padding 的自适应放大策略 ——
        self.max_pad_bumps_ctx  = int(cfg.get("max_pad_bumps_ctx", 16))   # 最多抬 context 次数
        self.max_pred_bumps     = int(cfg.get("max_pred_bumps", 24))      # 最多放大 pred_len 次数
        self.pred_bump_growth   = float(cfg.get("pred_bump_growth", 1.75))# 每次放大倍率
        self.pred_len_cap       = int(cfg.get("pred_len_cap", 2048))      # 放大上限
        self.max_retries        = int(cfg.get("max_retries", 200))        # 总重试上限

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

    def _build_estimator(self,
                         prediction_length:int,
                         context_length:int,
                         freq:str,
                         num_parallel_samples:int,
                         force_input_size: Optional[int] = None):
        Estimator = _resolve_estimator_cls()

        # 单变量时把 mg 组等裁剪到 1 组，避免 split 报错
        if (self.input_size or 1) == 1 and self.num_gran > 1:
            print(f"[mgtsd_adapter] 单变量输入(input_size=1) -> 将 num_gran {self.num_gran} 降为 1，并裁剪 mg/share/weight 列表")
            self.num_gran = 1
            if len(self.mg_list) > 1:     self.mg_list = self.mg_list[:1]
            if len(self.share_list) > 1:  self.share_list = self.share_list[:1]
            if len(self.weight_list) > 1: self.weight_list = self.weight_list[:1]

        core_kwargs = dict(
            prediction_length=prediction_length,
            context_length=context_length,
            freq=freq,
            input_size=int(force_input_size if force_input_size is not None else (self.input_size or 1)),
            target_dim=int(self.target_dim or 1),
        )
        if _sig_accepts(Estimator.__init__, "num_gran"):
            core_kwargs["num_gran"] = self.num_gran
        ctor_kwargs = _filter_kwargs(Estimator.__init__, core_kwargs)
        est = Estimator(**ctor_kwargs)
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
        # 保证：max_lag + ctx + pred + safety
        ctx_eff = max(ctx, self.min_ctx_floor)
        return max_lag + ctx_eff + pred + self.safety_margin

    def _extend_left(self, arr: np.ndarray, need_len: int) -> np.ndarray:
        L = arr.shape[-1]
        if L >= need_len:
            return arr.astype(float)
        pad = need_len - L
        if arr.ndim == 1:
            pad_val = float(arr[0]) if L > 0 else 0.0
            left = np.full((pad,), pad_val, dtype=float)
            return np.concatenate([left, arr.astype(float)], axis=0)
        else:
            pad_val = arr[:, 0:1] if L > 0 else np.zeros((arr.shape[0],1), dtype=float)
            left = np.repeat(pad_val, pad, axis=1)
            return np.concatenate([left, arr.astype(float)], axis=1)

    def _normalize_items(self,
                         raw_items: Iterable[Dict[str, Any]],
                         freq: str,
                         expect_2d: bool,
                         ctx: int,
                         pred: int,
                         max_lag: int) -> List[Dict[str, Any]]:
        need_len = self._min_required_len(ctx, pred, max_lag)
        out = []
        for it in raw_items:
            tgt = it["target"]
            if expect_2d:
                arr = _as_2d_target(tgt)
            else:
                arr = _as_1d_target(tgt)
            arr = self._extend_left(arr, need_len)
            out.append({"target": arr, "start": _to_period(it.get("start", "2000-01-01"), freq)})
        return out

    # ---------- 训练主流程（只改适配器） ----------

    def _train_once(self, estimator, train_ds):
        train_kwargs = _filter_kwargs(estimator.train, dict(training_data=train_ds))
        return estimator.train(**train_kwargs)

    def _train_with_retries(
        self,
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
        cur_ctx = int(context_length)
        cur_force_input_size: Optional[int] = None

        # 训练用的 prediction_length（只影响训练/卷积 padding，推理仍按原 pred 截断）
        train_pred = int(prediction_length)

        pad_bumps = 0
        pred_bumps = 0

        while True:
            tries += 1
            if tries > self.max_retries:
                raise RuntimeError(f"[mgtsd_adapter] 超过最大重试次数 {self.max_retries}，请检查配置或模型参数。")

            # 基于当前策略重建数据集（只做数据侧兜底）
            norm = self._normalize_items(items, freq, cur_expect_2d, cur_ctx, train_pred, cur_max_lag)
            ds = (_RawListDataset if cur_expect_2d else ListDataset)(norm, freq=freq)

            try:
                est = self._build_estimator(
                    prediction_length=train_pred,
                    context_length=cur_ctx,
                    freq=freq,
                    num_parallel_samples=prediction_length,  # 推理仍按原配置采样步数
                    force_input_size=cur_force_input_size,
                )
                # 明确设置 lags
                for k in ("lags_seq", "lags", "lag_indices", "lag_seq"):
                    if hasattr(est, k):
                        setattr(est, k, [1] if cur_max_lag <= 1 else list(range(1, cur_max_lag + 1)))

                pred = self._train_once(est, ds)
                self._trained_expect_2d = cur_expect_2d
                return pred, cur_expect_2d, cur_max_lag

            except GluonTSDataError as ge:
                msg = str(ge)
                # 1D/2D 自动切换
                if _need_2d(msg) and not cur_expect_2d:
                    cur_expect_2d = True
                    continue
                if _need_1d(msg) and cur_expect_2d:
                    cur_expect_2d = False
                    continue
                raise  # 其它 GluonTSDataError 无法兜底

            except AssertionError as ae:
                text = str(ae)
                lag_hist = _is_lag_hist_assert(text)
                if lag_hist is not None:
                    found_lag, hist = lag_hist
                    allowed_ctx = hist - prediction_length - max(found_lag, cur_max_lag)
                    new_ctx = max(self.min_ctx_floor, min(cur_ctx, allowed_ctx))
                    if new_ctx < cur_ctx:
                        print(f"[mgtsd_adapter] 解析到 history={hist}, max_lag={max(found_lag, cur_max_lag)} -> "
                              f"自动下调 context_length: {cur_ctx} -> {new_ctx}")
                        cur_ctx = new_ctx
                    if cur_max_lag != 1:
                        print(f"[mgtsd_adapter] 将 lags 收敛到 [1]（原最大滞后 {cur_max_lag}）")
                    cur_max_lag = 1
                    continue
                raise

            except RuntimeError as re_err:
                text = str(re_err)

                # RNN input_size 不匹配 -> 解析 "Expected x, got y" 并重建 Estimator
                got = _parse_input_size_mismatch(text)
                if got is not None and (cur_force_input_size or (self.input_size or 1)) != got:
                    print(f"[mgtsd_adapter] 侦测到 input_size 不匹配：Estimator.input_size="
                          f"{(cur_force_input_size or self.input_size or 1)}, 实际特征维={got} "
                          f"-> 重建 Estimator(input_size={got}) 并重训")
                    cur_force_input_size = got
                    continue

                # —— 关键：circular padding —— 优先放大“训练用 prediction_length”，确保 T > padding
                if "Padding value causes wrapping around more than once" in text:
                    # 先尝试放大 pred，再考虑抬 context（抬 context 对这个报错通常无效）
                    if pred_bumps < self.max_pred_bumps and train_pred < self.pred_len_cap:
                        new_pred = int(min(max(train_pred + 1, np.ceil(train_pred * self.pred_bump_growth)), self.pred_len_cap))
                        print(f"[mgtsd_adapter] circular padding 仍存在 -> 放大训练用 prediction_length: {train_pred} -> {new_pred} (推理仍按 {prediction_length} 截取)")
                        train_pred = new_pred
                        pred_bumps += 1
                        continue

                    # 次优先：抬 context 下限（针对编码端其它层）
                    if pad_bumps < self.max_pad_bumps_ctx:
                        new_floor = self.min_ctx_floor + 96 if self.min_ctx_floor < 256 else self.min_ctx_floor + 128
                        print(f"[mgtsd_adapter] circular padding 仍存在 -> 提高最小上下文下限: {self.min_ctx_floor} -> {new_floor}")
                        self.min_ctx_floor = new_floor
                        cur_ctx = max(cur_ctx, self.min_ctx_floor)
                        pad_bumps += 1
                        continue

                    # 兜底失败：给出明确提示
                    raise RuntimeError(
                        f"[mgtsd_adapter] circular padding 仍无法满足："
                        f"已抬 context {pad_bumps} 次、放大训练用 pred_len {pred_bumps} 次(当前 {train_pred})。"
                        f"请检查卷积核/扩张设定，或在外部 config 提高 prediction_length（建议 >= {train_pred*2 if train_pred<512 else train_pred}）/context_length。"
                    )

                # 其它 RuntimeError 抛出
                raise

    def fit_or_load(self,
                    train_items: Iterable[Dict[str, Any]],
                    freq: str,
                    prediction_length: int,
                    context_length: int,
                    num_samples_for_predictor: int):
        _ensure_wandb_stub()

        items = list(train_items)

        # 尺度推断
        if self.input_size is None or self.target_dim is None:
            i_size, t_dim = self._infer_dims_from_items(items)
            if self.input_size is None: self.input_size = i_size
            if self.target_dim is None: self.target_dim = t_dim
        self.input_size = int(self.input_size or 1)
        self.target_dim = int(self.target_dim or 1)

        # 单变量裁剪多粒度
        if self.input_size == 1 and self.num_gran > 1:
            print(f"[mgtsd_adapter] 单变量输入(input_size=1) -> 将 num_gran {self.num_gran} 降为 1，并裁剪 mg/share/weight 列表")
            self.num_gran = 1
            if len(self.mg_list) > 1:     self.mg_list = self.mg_list[:1]
            if len(self.share_list) > 1:  self.share_list = self.share_list[:1]
            if len(self.weight_list) > 1: self.weight_list = self.weight_list[:1]

        # 两条路：先从 1D 试，失败再 2D
        try:
            pred, used_2d, used_max_lag = self._train_with_retries(
                items=items, freq=freq,
                prediction_length=prediction_length,
                context_length=context_length,
                expect_2d=False,
                max_lag=self.max_lag_guess
            )
            self.predictor = pred
            self._trained_expect_2d = used_2d
            return self
        except Exception:
            pred, used_2d, used_max_lag = self._train_with_retries(
                items=items, freq=freq,
                prediction_length=prediction_length,
                context_length=context_length,
                expect_2d=True,
                max_lag=self.max_lag_guess
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
                    rep = int(np.ceil(num_samples / max(1, S.shape[0]))); S = np.concatenate([S] * rep, axis=0)[:num_samples, :]
            return S

        # 推理侧也仅做数据兜底，不动模型
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
    # 优先走统一入口（若提供）
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

    # 否则逐项尝试设置属性（容错）
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
