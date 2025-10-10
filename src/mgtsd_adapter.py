# /home/shihongyue/MG-TSD-operation/src/mgtsd_adapter.py
from __future__ import annotations
import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Iterable
import numpy as np
import torch

from gluonts.dataset.common import ListDataset

def _resolve_estimator_cls():
    """
    尝试从 mgtsd_estimator / estimator 里找到 Estimator 类（带 .train(...) 返回 PyTorchPredictor）。
    """
    mod_names = ["mgtsd_estimator", "estimator"]
    cand_names = ["MGTSDEstimator", "mgtsdEstimator", "Estimator", "MGTSD_Estimator"]
    last_err = None
    for mn in mod_names:
        try:
            m = importlib.import_module(mn)
        except Exception as e:
            last_err = e
            continue
        for cn in cand_names:
            est = getattr(m, cn, None)
            if isinstance(est, type) and hasattr(est, "train"):
                return est
    raise ImportError(f"找不到 Estimator 类；请确认 src/mgtsd_estimator.py 中对外暴露的类名（最后错误：{last_err})")

class MGAdapter:
    """
    统一接口：
      - fit_or_load(train_items, freq, prediction_length, context_length, num_samples)
      - predict_one(x_ctx, pred_len, num_samples) -> (num_samples, pred_len)
    """
    def __init__(self, cfg: Dict[str, Any], device: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.predictor = None
        # 超参（与仓库命名对齐）
        self.epoch       = int(cfg.get("epoch", 30))
        self.diff_steps  = int(cfg.get("diff_steps", 100))
        self.batch_size  = int(cfg.get("batch_size", 128))
        self.num_cells   = int(cfg.get("num_cells", 64))
        self.mg_dict     = str(cfg.get("mg_dict", "1_4_12"))
        self.num_gran    = int(cfg.get("num_gran", 3))
        self.share_ratio = str(cfg.get("share_ratio_list", "1_0.8_0.6"))
        self.weight_list = str(cfg.get("weight_list", "0.8_0.1_0.1"))
        self.ckpt_path   = Path(cfg.get("ckpt_path", "")).expanduser()

    def _build_estimator(self,
                         prediction_length:int,
                         context_length:int,
                         freq:str,
                         num_parallel_samples:int):
        Estimator = _resolve_estimator_cls()

        # 有些实现的 __init__ 参数名会略有出入；准备两套参数，逐一尝试
        strict = dict(
            prediction_length=prediction_length,
            context_length=context_length,
            freq=freq,
            batch_size=self.batch_size,
            epochs=self.epoch,
            num_parallel_samples=num_parallel_samples,
            diff_steps=self.diff_steps,
            num_cells=self.num_cells,
            mg_dict=self.mg_dict,
            num_gran=self.num_gran,
            share_ratio_list=self.share_ratio,
            weight_list=self.weight_list,
            device=str(self.device),
        )
        minimal = dict(
            prediction_length=prediction_length,
            context_length=context_length,
            freq=freq,
            batch_size=self.batch_size,
            epochs=self.epoch,
            num_parallel_samples=num_parallel_samples,
        )
        tried = []
        for params in (strict, minimal):
            try:
                return Estimator(**params)
            except TypeError as e:
                tried.append((params.keys(), e))
                continue
        raise TypeError(f"实例化 Estimator 失败。尝试的参数集合：{tried}")

    def _to_listdataset(self, items: Iterable[Dict[str, Any]], freq: str) -> ListDataset:
        # 保证每个 item 至少含 target/start
        def _norm(it):
            return {"target": it["target"], "start": it.get("start", "2000-01-01")}
        return ListDataset((_norm(it) for it in items), freq=freq)

    def fit_or_load(self,
                    train_items: Iterable[Dict[str, Any]],
                    freq: str,
                    prediction_length: int,
                    context_length: int,
                    num_samples_for_predictor: int):
        """
        训练或加载 predictor。
        - train_items: 直接用 GluonTS JSON 里的条目（有 target/start）
        - num_samples_for_predictor: 作为 estimator 的 num_parallel_samples
        """
        # 先构造 Estimator
        estimator = self._build_estimator(prediction_length, context_length, freq, num_samples_for_predictor)

        # （可选）如果你实现了从 ckpt 复现 predictor，可在这里走加载分支；
        # 多数 GluonTS Torch 模型直接 estimator.train(...) 会更省事、稳定。
        train_ds = self._to_listdataset(train_items, freq=freq)
        self.predictor = estimator.train(training_data=train_ds)  # 返回 PyTorchPredictor
        return self

    @torch.no_grad()
    def predict_one(self, x_ctx: np.ndarray, pred_len: int, num_samples: int, freq: str = "B") -> np.ndarray:
        """
        单序列预测（GluonTS PyTorchPredictor）：
        - 构造一个只含上下文的 ListDataset 条目，predictor 会用内部的 InstanceSplitter 截取最后 context 区间
        """
        assert self.predictor is not None, "predictor 未初始化，请先调用 fit_or_load(...)"
        # 构造单条样本
        test_ds = ListDataset([{"target": x_ctx.astype(float).tolist(), "start": "2000-01-01"}], freq=freq)
        it = self.predictor.predict(test_ds)
        fc = next(iter(it))  # Forecast
        S = np.asarray(fc.samples)  # 形状可能是 (S,L) 或 (S,L,D) / (S,L,G,D)
        # 只保留第一个通道，保证输出 (S, L)
        if S.ndim == 2:
            pass
        elif S.ndim >= 3:
            S = S[..., 0]  # 取第 1 个 target 维/粒度
        else:
            raise RuntimeError(f"未知的 samples 维度: {S.shape}")
        # 截断/校正到期望的 pred_len 和 num_samples
        if S.shape[1] != pred_len:
            S = S[:, :pred_len]
        if S.shape[0] != num_samples:
            # 若内部并行样本数与要求不一致，做简易重采样（重复或裁剪）
            if S.shape[0] > num_samples:
                S = S[:num_samples, :]
            else:
                rep = int(np.ceil(num_samples / S.shape[0]))
                S = np.concatenate([S] * rep, axis=0)[:num_samples, :]
        return S
