import chainer
import chainer.functions as F

from chainer import Variable


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.cae = kwargs.pop('model')
        super(Updater, self).__init__(*args, **kwargs)

    def loss(self, cae, x, t):
        loss = F.mean_squared_error(x, t)
        chainer.report({'loss': loss}, cae)
        return loss

    def update_core(self):
        optimizer = self.get_optimizer('cae')
        cae = self.cae
        xp = cae.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        in_ch = batch[0][0].shape[0]
        out_ch = batch[0][1].shape[0]

        w_in = batch[0][0].shape[1]
        w_out = batch[0][1].shape[1]

        x_in = xp.zeros((batchsize, in_ch, w_in, w_in)).astype("f")
        t = xp.zeros((batchsize, out_ch, w_out, w_out)).astype("f")

        for i in range(batchsize):
            x_in[i, :] = xp.asarray(batch[i][0])
            t[i, :] = xp.asarray(batch[i][1])

        x_in = Variable(x_in)
        x_out = cae(x_in)
        optimizer.update(self.loss, cae, x_out, t)
