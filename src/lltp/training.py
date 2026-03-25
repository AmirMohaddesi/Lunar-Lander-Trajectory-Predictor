"""Training steps matching the reference notebook (batch size 1 for RNN loop)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def ae_loss(y: torch.Tensor, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    del z
    x = x.reshape(*y.shape)
    return F.mse_loss(y, x, reduction="mean")


def ae_train_step(
    x: torch.Tensor,
    net: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
) -> torch.Tensor:
    net.train()
    z = net.encoder(x)
    y = net.decoder(z)
    loss = loss_fn(y, x, z)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.detach()


def ae_eval_step(x: torch.Tensor, net: nn.Module, loss_fn) -> torch.Tensor:
    net.eval()
    with torch.no_grad():
        z = net.encoder(x)
        y = net.decoder(z)
        return loss_fn(y, x, z).detach()


def latent_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def rnn_train_step(
    x_latent: torch.Tensor,
    t_latent: torch.Tensor,
    rnn: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ``x_latent`` is expected with shape [batch, dimz] (batch=1 in the original notebook).
    The outer loop over time steps iterates the batch dimension (notebook behavior).
    """
    rnn.train()
    batch_size = x_latent.shape[0]
    ht = rnn.init_state(batch_size, device=x_latent.device)
    y = x_latent
    for xt in x_latent:
        y, ht = rnn(xt, ht)
    loss = loss_fn(y, t_latent)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return y.detach(), loss.detach()


def rnn_eval_step(
    x_latent: torch.Tensor,
    t_latent: torch.Tensor,
    rnn: nn.Module,
    loss_fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    rnn.eval()
    batch_size = x_latent.shape[0]
    with torch.no_grad():
        ht = rnn.init_state(batch_size, device=x_latent.device)
        y = x_latent
        for xt in x_latent:
            y, ht = rnn(xt, ht)
        loss = loss_fn(y, t_latent)
        return y, loss.detach()
