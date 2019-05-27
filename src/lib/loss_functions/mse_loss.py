from chainer import functions as F


def mse_loss(x, t):
    return F.mean_squared_error(x, t)
