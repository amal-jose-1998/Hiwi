"""
Network-1: Auto-Decoder SDF model.
Learns:
    f_theta(z_i, xyz) -> sdf
Latent codes z_i are stored in an nn.Embedding table and are optimized jointly with the decoder.
"""

import torch
import torch.nn as nn

class Network1SDF(nn.Module):
    """
    Auto-decoder SDF network.

    Parameters
    num_geometries : int
        Number of geometries K. This sets the size of the latent embedding table.
    latent_dim : int
        Dimension of each latent code z_i.
    hidden_dim : int
        Width of the MLP hidden layers.
    depth : int
        Number of linear layers in the decoder MLP including the final output layer.
        Must be >= 2 to have at least one hidden layer + output layer.

    Forward
    Inputs:
        geom_id : LongTensor of shape (B,)
            Geometry IDs in [0, K-1].
        xyz : FloatTensor of shape (B, 3)
            Query points.

    Output:
        sdf_pred : FloatTensor of shape (B, 1)
            Predicted signed distance values.
    """

    def __init__(self, num_geometries, latent_dim=128, hidden_dim=256, depth=4, latent_init_std=0.012, activation="relu", use_skip=True, skip_layer=None,):
        super().__init__()

        if depth < 2:
            raise ValueError(f"depth must be >= 2 (got {depth})")

        self.num_geometries = int(num_geometries)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)

        # Latent embedding table: one learnable vector per geometry
        self.latent = nn.Embedding(self.num_geometries, self.latent_dim)
        nn.init.normal_(self.latent.weight, mean=0.0, std=float(latent_init_std))

        if activation == "relu":
            act = nn.ReLU(inplace=True)
        elif activation == "softplus":
            act = nn.Softplus(beta=100)  
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.use_skip = bool(use_skip)
        num_hidden = depth - 1  # hidden Linear count
        if self.use_skip:
            if skip_layer is None:
                skip_layer = max(0, (num_hidden // 2) - 1)
            self.skip_layer = int(skip_layer)
        else:
            self.skip_layer = -1

        self.fcs = nn.ModuleList()
        self.acts = nn.ModuleList()

        in_dim = self.latent_dim + 3

        for h in range(num_hidden):
            # If we inject skip AFTER layer h, we need next layer to accept extra inputs.
            self.fcs.append(nn.Linear(in_dim, hidden_dim))
            self.acts.append(act)
            in_dim = hidden_dim

            # After this hidden layer, if it's the skip point, the next layer input will grow
            if self.use_skip and h == self.skip_layer:
                in_dim = hidden_dim + (self.latent_dim + 3)

        self.out = nn.Linear(in_dim, 1)

    @classmethod
    def from_cfg(cls, num_geometries, cfg):
        """
        Convenience constructor to build the model using Network1Config.
        """
        return cls(
            num_geometries=num_geometries,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            depth=cfg.depth,
            latent_init_std=getattr(cfg, "latent_init_std", 0.012),
            activation=getattr(cfg, "activation", "relu"),
            use_skip=getattr(cfg, "use_skip", True),
            skip_layer=getattr(cfg, "skip_layer", None),
        )
    
    def forward(self, geom_id, xyz):
        """
        Forward pass.

        Parameters
        geom_id : torch.Tensor
            Long tensor of shape (B,). Values must be in [0, num_geometries-1].
        xyz : torch.Tensor
            Float tensor of shape (B, 3).

        Returns
        torch.Tensor
            Float tensor of shape (B, 1) with predicted SDF values.
        """
        z = self.latent(geom_id)  # (B, L)
        inp = torch.cat([z, xyz], dim=1)  # (B, L+3)

        h = inp
        for i, (fc, act) in enumerate(zip(self.fcs, self.acts)):
            h = act(fc(h))
            if self.use_skip and i == self.skip_layer:
                h = torch.cat([h, inp], dim=1)

        return self.out(h)

    def latent_codes(self):
        """
        Return the full latent embedding table of shape (K, latent_dim).

        Useful for:
        - latent L2 regularization during training
        - saving/exporting latent codes after training
        """
        return self.latent.weight
    
    def forward_with_latent(self, z, xyz):
        if z.dim() == 1:
            z = z[None, :].expand(xyz.size(0), -1)
        inp = torch.cat([z, xyz], dim=1)

        h = inp
        for i, (fc, act) in enumerate(zip(self.fcs, self.acts)):
            h = act(fc(h))
            if self.use_skip and i == self.skip_layer:
                h = torch.cat([h, inp], dim=1)

        return self.out(h)