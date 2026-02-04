import torch
import pytorch_lightning as pl

from .model import Network1SDF


def sdf_loss_l1(pred, target, delta):
    """
    It computes L1 error between predicted SDF and true SDF. But before computing error, it truncates both values 
    to [-delta, +delta], for learning the surface region accurately.
    """
    pred_c = torch.clamp(pred, -delta, delta)
    tgt_c = torch.clamp(target, -delta, delta)
    return torch.mean(torch.abs(pred_c - tgt_c))


def latent_l2_regularizer(model, pair_id):
    """
    L2 regularization on latent codes actually used in the batch.
    """
    unique_ids = torch.unique(pair_id) # Finds which latent codes were used in this batch
    z = model.latent(unique_ids)
    return torch.mean(z ** 2) # Applies an L2 penalty (mean squared magnitude) on those latent vectors only.


class Network1LitModule(pl.LightningModule):
    def __init__(self, cfg, num_train_geoms, stats):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg", "stats"])
        self.cfg = cfg

        # number of components (die / punch / binder)
        self.num_components = len(getattr(cfg, "tool_components", ["die", "punch", "binder"]))

        self.sdf_scale = float(stats.get("sdf_scale", 1.0))
        if self.sdf_scale == 0:
            self.sdf_scale = 1.0

        # Build the real network
        self.model = Network1SDF(
            num_geometries=num_train_geoms,
            num_components=self.num_components,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            depth=cfg.depth,
            latent_init_std=getattr(cfg, "latent_init_std", 0.012),
            activation=getattr(cfg, "activation", "relu"),
            use_skip=getattr(cfg, "use_skip", True),
            skip_layer=getattr(cfg, "skip_layer", None),
        )

    def forward(self, pair_id, xyz):
        return self.model(pair_id, xyz)

    def _shared_step(self, batch, stage): # used for both train and val.
        # unpack batch 
        geom_id, comp_id, xyz, sdf = batch

        # compute (geom, comp) pair id 
        pair_id = geom_id * self.num_components + comp_id

        # forward pass
        pred = self(pair_id, xyz) # This calls Network1SDF.forward()

        # losses
        delta = float(getattr(self.cfg, "truncation_delta", 0.05))
        sdf_l = sdf_loss_l1(pred, sdf, delta=delta) # Reconstruction loss (clamped L1)
        reg_l = latent_l2_regularizer(self.model, pair_id) # Latent regularization

        loss = sdf_l + float(self.cfg.latent_l2_weight) * reg_l 

        # logging
        if stage == "train":
            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/sdf_l1_clamped", sdf_l, on_step=True, on_epoch=True)
            self.log("train/latent_reg", reg_l, on_step=True, on_epoch=True)
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
        """
        optimizes:
            decoder MLP weights
            latent embedding weights
        """
        opt = torch.optim.Adam(
            self.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

        step_size = int(getattr(self.cfg, "lr_step_size_epochs", 50))
        gamma = float(getattr(self.cfg, "lr_gamma", 0.5))
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma) # Scheduler: Every step_size epochs, learning rate gets multiplied by gamma.

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_start(self):
        # to make the BalancedGeomBatchSampler change its RNG each epoch. so this ensures that each epoch gets a new shuffled sampling pattern.
        dl = self.trainer.train_dataloader
        if hasattr(dl, "batch_sampler") and hasattr(dl.batch_sampler, "set_epoch"):
            dl.batch_sampler.set_epoch(self.current_epoch)

    def on_train_epoch_end(self):
        # prevents StepLR from shrinking the LR below a minimum.
        lr_min = float(getattr(self.cfg, "lr_min", 5e-6))
        for pg in self.trainer.optimizers[0].param_groups:
            if pg["lr"] < lr_min:
                pg["lr"] = lr_min
