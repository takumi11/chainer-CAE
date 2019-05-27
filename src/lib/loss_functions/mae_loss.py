from chainer import functions as F


def mae_loss(x, t):
    return F.mean_absolute_error(x, t)
