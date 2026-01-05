import torch
import pytorch_lightning as pl

from .model import Network1SDF


def sdf_loss_l1(pred, target, delta):
    """
    L1 between clamped(pred) and clamped(target).
    delta corresponds to truncation distance.
    """
    pred_c = torch.clamp(pred, -delta, delta)
    tgt_c = torch.clamp(target, -delta, delta)
    return torch.mean(torch.abs(pred_c - tgt_c))


def latent_l2_regularizer(model, geom_id):
    unique_ids = torch.unique(geom_id)
    z = model.latent(unique_ids)
    return torch.mean(z ** 2)


class Network1LitModule(pl.LightningModule):
    def __init__(self, cfg, num_train_geoms, stats):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg", "stats"])
        self.cfg = cfg

        self.sdf_scale = float(stats.get("sdf_scale", 1.0))
        if self.sdf_scale == 0:
            self.sdf_scale = 1.0

        self.model = Network1SDF(
            num_geometries=num_train_geoms,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            depth=cfg.depth,
            latent_init_std=getattr(cfg, "latent_init_std", 0.012),
            activation=getattr(cfg, "activation", "relu"),
        )

    def forward(self, geom_id, xyz):
        return self.model(geom_id, xyz)
    
    def _shared_step(self, batch, stage):
        geom_id, xyz, sdf = batch
        pred = self(geom_id, xyz)

        delta = float(getattr(self.cfg, "truncation_delta", 0.05))
        #delta = delta / max(self.sdf_scale, 1e-12)
        sdf_l = sdf_loss_l1(pred, sdf, delta=delta)

        reg_l = latent_l2_regularizer(self.model, geom_id)
        loss = sdf_l + float(self.cfg.latent_l2_weight) * reg_l

        if stage == "train":
            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/sdf_l1_clamped", sdf_l, prog_bar=False, on_step=True, on_epoch=True)
            self.log("train/latent_reg", reg_l, prog_bar=False, on_step=True, on_epoch=True)
        else:
            self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val/sdf_l1_clamped", sdf_l, on_step=False, on_epoch=True)
            self.log("val/latent_reg", reg_l, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )
        step_size = int(getattr(self.cfg, "lr_step_size_epochs", 50))
        gamma = float(getattr(self.cfg, "lr_gamma", 0.5))
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
    def on_train_epoch_start(self):
        dl = self.trainer.train_dataloader
        if hasattr(dl, "batch_sampler") and hasattr(dl.batch_sampler, "set_epoch"):
            dl.batch_sampler.set_epoch(self.current_epoch)

    def on_train_epoch_end(self):
        lr_min = float(getattr(self.cfg, "lr_min", 5e-6))
        for pg in self.trainer.optimizers[0].param_groups:
            if pg["lr"] < lr_min:
                pg["lr"] = lr_min