#!/usr/bin/env python
"""MeshMAE 風の自己教師あり学習エントリーポイント（簡略版）。"""
import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


class DummyMeshDataset(Dataset):
    """メッシュが存在しない環境でも動作させるための簡易データセット。"""

    def __init__(self, length: int = 16, feature_dim: int = 1024):
        self.length = length
        self.feature_dim = feature_dim

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        _ = idx
        return torch.randn(self.feature_dim), torch.randn(self.feature_dim)


def build_model(feature_dim: int = 1024) -> nn.Module:
    """本家モデルの代替として単純な自己再構成ネットワークを構築する。"""

    return nn.Sequential(
        nn.Linear(feature_dim, feature_dim // 2),
        nn.ReLU(inplace=True),
        nn.Linear(feature_dim // 2, feature_dim),
    )


def load_state_dict_flexible(model: nn.Module, state_dict_container):
    """state_dict が多様なキーで格納されている可能性に対応して読み込む。"""

    if isinstance(state_dict_container, dict):
        if "model" in state_dict_container:
            state_dict = state_dict_container["model"]
        elif "state_dict" in state_dict_container:
            state_dict = state_dict_container["state_dict"]
        else:
            state_dict = state_dict_container
    else:
        state_dict = state_dict_container

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    return missing_keys, unexpected_keys


def parse_args():
    parser = argparse.ArgumentParser(description="MeshMAE ドメイン適応学習")
    parser.add_argument("--dataroot", default="./datasets/", help="データセットルート")
    parser.add_argument("--batch_size", type=int, default=8, help="バッチサイズ")
    parser.add_argument("--epochs", type=int, default=20, help="学習エポック数")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="マスク率（ダミー）")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="学習率")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight Decay")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="使用デバイス (cuda / cpu)",
    )
    parser.add_argument("--init", default="", help="初期化に利用するチェックポイント")
    parser.add_argument("--resume", default="", help="学習再開に利用するチェックポイント")
    parser.add_argument("--save_ckpt", default="", help="保存先チェックポイントパス")
    return parser.parse_args()


@contextmanager
def maybe_autocast(enabled: bool):
    """AMP を使用するかどうかを切り替えるコンテキスト。"""

    if enabled:
        with autocast():
            yield
    else:
        yield


def main():
    args = parse_args()

    device = torch.device(args.device)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")

    missing_keys = []
    unexpected_keys = []
    start_epoch = 0

    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"再開用チェックポイントが見つかりません: {ckpt_path}")
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        missing_keys, unexpected_keys = load_state_dict_flexible(model, checkpoint)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0))
        print(
            f"Resume from {ckpt_path}: missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}, start_epoch={start_epoch}"
        )
    elif args.init:
        init_path = Path(args.init)
        if not init_path.exists():
            raise FileNotFoundError(f"初期化用チェックポイントが見つかりません: {init_path}")
        checkpoint = torch.load(str(init_path), map_location="cpu")
        missing_keys, unexpected_keys = load_state_dict_flexible(model, checkpoint)
        print(
            f"Init from {init_path}: missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
        )
    else:
        print("ランダム初期化で学習を開始します。")

    dataset = DummyMeshDataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        start_time = time.time()
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            with maybe_autocast(scaler.is_enabled()):
                outputs = model(inputs)
                loss = nn.functional.mse_loss(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * inputs.size(0)

        epoch_loss /= len(dataset)
        duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {epoch_loss:.6f} ({duration:.2f}s)")

        if args.save_ckpt:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
            }
            torch.save(ckpt, args.save_ckpt)

    if args.save_ckpt:
        print(f"学習済みモデルを保存しました: {args.save_ckpt}")

    summary = {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
