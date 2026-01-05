import random
import numpy as np
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from .config import Network1Config
from .data import build_sdf_dataloader  
from .trainer import Network1LitModule

from pathlib import Path


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    cfg = Network1Config()
    set_seed(cfg.seed)

    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, n_train_geoms, n_test_geoms, stats = build_sdf_dataloader(cfg)
    print(f"[train_network1] train_geoms={n_train_geoms}, test_geoms={n_test_geoms}")

    lit_model = Network1LitModule(cfg=cfg, num_train_geoms=n_train_geoms, stats=stats)

    early_cb = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=10,      
        min_delta=0.0,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=cfg.ckpt_dir,
        monitor="val/loss",
        filename="network1-{epoch:03d}-{val_loss:.4f}",
        save_top_k=3,              
        every_n_epochs=cfg.save_every,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="gpu" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu",
        devices=1,
        callbacks=[ckpt_cb, early_cb],
        log_every_n_steps=2,
    )

    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    print("[train_network1] Done.")


if __name__ == "__main__":
    main()