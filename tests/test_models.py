import torch

from lltp.models import ConvAE, SimpleRNN


def test_conv_ae_roundtrip_shape():
    net = ConvAE(dimz=50)
    x = torch.randn(2, 3, 64, 64)
    y = net(x)
    assert y.shape == x.shape
    z = net.encoder(x)
    assert z.shape == (2, 50)


def test_simple_rnn_one_step():
    rnn = SimpleRNN(latent_dim=50, nhid=200)
    b = 1
    z = torch.randn(b, 50)
    h = rnn.init_state(b)
    y, h2 = rnn(z, h)
    assert y.shape == (b, 50)
    assert h2.shape == (b, 200)


def test_training_rnn_loop_matches_notebook_batch1():
    """Notebook iterates ``for xt in x_latent`` with batch matrix [1, 50]."""
    rnn = SimpleRNN(latent_dim=50, nhid=200)
    x_latent = torch.randn(1, 50)
    h = rnn.init_state(1)
    y = x_latent
    for xt in x_latent:
        y, h = rnn(xt, h)
    assert y.shape == (1, 50)
