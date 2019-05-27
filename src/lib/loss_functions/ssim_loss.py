from chainer import functions as F


def ssim_loss(x, t):

    batch_size, ch, h, w = x.shape

    loss = 0.0
    for i in range(batch_size):
        loss += SSIM(x[i], t[i])

    return loss / batch_size


def SSIM(x, t, k1=0.01, k2=0.03):

    C1 = k1**2
    C2 = k2**2

    x_mean = F.mean(x)
    t_mean = F.mean(t)

    x_var = F.mean(x * x) - x_mean**2
    t_var = F.mean(t * t) - t_mean**2
    cov = F.mean(x * t) - x_mean * t_mean

    a = (2 * x_mean * t_mean + C1) * (2 * cov + C2)
    b = (x_mean**2 + t_mean**2 + C1) * (x_var + t_var + C2)
    ssim = a / b

    return 1 - ssim
