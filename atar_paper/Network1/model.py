"""
Network-1: Auto-Decoder SDF model.

Learns:
    f_theta(z_{geom, comp}, xyz) = sdf

    where,
        xyz = a 3D point in space
        z_(geom,comp) = a learned latent vector that represents one specific tool geometry + one specific component
        output = the signed distance at that point (sdf)

Latent codes are defined per (geometry, component) pair.
"""

import torch
import torch.nn as nn

class Network1SDF(nn.Module):
    """
    Auto-decoder SDF network.

    Parameters
    num_geometries : int
        Number of unique geometries K.
    num_components : int
        Number of tool components C (e.g. die, punch, binder).
    latent_dim : int
        Dimension of each latent code.
    hidden_dim : int
        Width of hidden layers.
    depth : int
        Number of MLP layers including output.
    """

    def __init__(
        self,
        num_geometries,
        num_components,
        latent_dim=128,
        hidden_dim=256,
        depth=4,
        latent_init_std=0.012,
        activation="relu",
        use_skip=True,
        skip_layer=None,
        softplus_beta=20.0,
    ):
        super().__init__()

        if depth < 2:
            raise ValueError("depth must be >= 2")

        self.num_geometries = int(num_geometries)
        self.num_components = int(num_components)
        self.num_pairs = self.num_geometries * self.num_components

        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)

        # --------------------------------------------------
        # Latent embedding: one per (geometry, component)
        # --------------------------------------------------
        self.latent = nn.Embedding(self.num_pairs, self.latent_dim) # trainable lookup table
        nn.init.normal_(self.latent.weight, mean=0.0, std=float(latent_init_std))

        # --------------------------------------------------
        # Activation
        # --------------------------------------------------
        if activation == "relu":
            make_act = lambda: nn.ReLU(inplace=False)
        elif activation == "softplus":
            make_act = lambda: nn.Softplus(beta=float(softplus_beta))
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.use_skip = bool(use_skip)
        num_hidden = depth - 1

        if self.use_skip:
            if skip_layer is None:
                skip_layer = max(0, (num_hidden // 2) - 1)
            self.skip_layer = int(skip_layer)
        else:
            self.skip_layer = -1

        # --------------------------------------------------
        # Decoder MLP
        # --------------------------------------------------
        self.fcs = nn.ModuleList() # Linear layers
        self.acts = nn.ModuleList() # activation layers

        base_in_dim = self.latent_dim + 3  # (z, xyz); Input dimension
        in_dim = base_in_dim

        for h in range(num_hidden):
            self.fcs.append(nn.Linear(in_dim, self.hidden_dim))
            self.acts.append(make_act())
            in_dim = self.hidden_dim

            if self.use_skip and h == self.skip_layer:
                in_dim = self.hidden_dim + base_in_dim

        self.out = nn.Linear(in_dim, 1) # SDF value.

    @classmethod
    def from_cfg(cls, num_geometries, cfg):
        return cls(
            num_geometries=num_geometries,
            num_components=len(cfg.tool_components),
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            depth=cfg.depth,
            latent_init_std=getattr(cfg, "latent_init_std", 0.012),
            activation=getattr(cfg, "activation", "relu"),
            use_skip=getattr(cfg, "use_skip", True),
            skip_layer=getattr(cfg, "skip_layer", None),
        )

    def forward(self, pair_id, xyz):
        """
        The normal training/inference path

        Parameters
        pair_id : LongTensor (B,)
            Encodes (geometry, component).
        xyz : FloatTensor (B, 3)

        Returns
        sdf : FloatTensor (B, 1)
        """
        z = self.latent(pair_id) # (B, latent_dim); lookup latent code
        inp = torch.cat([z, xyz], dim=1) # (B, latent_dim+3); concatenate latent and xyz

        # run MLP
        h = inp
        for i, (fc, act) in enumerate(zip(self.fcs, self.acts)):
            h = act(fc(h))
            if self.use_skip and i == self.skip_layer:
                h = torch.cat([h, inp], dim=1)

        return self.out(h) # (B,1)

    def latent_codes(self):
        """Return all latent vectors (num_pairs, latent_dim)."""
        return self.latent.weight # (num_pairs, latent_dim)
    
    def forward_with_latent(self, z, xyz) :
        """
        Evaluate decoder with an explicit latent vector z (no embedding lookup).
        Used in evaluation (latent optimization)

        z:   (D,) or (1,D)
        xyz: (N,3)
        """
        if z.dim() == 1:
            z = z[None, :]  # (1,D)
        z = z.expand(xyz.size(0), -1)  # (N,D)

        inp = torch.cat([z, xyz], dim=1)  # (N, D+3)

        h = inp
        for i, (fc, act) in enumerate(zip(self.fcs, self.acts)):
            h = act(fc(h))
            if self.use_skip and i == self.skip_layer:
                h = torch.cat([h, inp], dim=1)

        return self.out(h)

