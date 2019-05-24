#!/usr/bin/env python

import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable


def out_image(updater, gen, in_ch, out_ch, rows, cols, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp

        w_in = 256
        w_out = 256

        in_all = np.zeros((n_images, in_ch, w_in, w_in)).astype('f')
        cae_all = np.zeros((n_images, out_ch, w_out, w_out)).astype('f')

        for it in range(n_images):
            batch = updater.get_iterator('test').next()
            batchsize = len(batch)

            x_in = xp.zeros((batchsize, in_ch, w_in, w_in)).astype('f')
            t_out = xp.zeros((batchsize, out_ch, w_out, w_out)).astype('f')

            for i in range(batchsize):
                x_in[i, :] = xp.asarray(batch[i][0])
                t_out[i, :] = xp.asarray(batch[i][1])

            _x_in = Variable(x_in)
            with chainer.no_backprop_mode():
                with chainer.using_config('train', False):
                    x_out = gen(_x_in)

            in_all[it, :] = x_in.get()[0, :]
            cae_all[it, :] = x_out.array.get()[0, :]

        def save_image(x, name, mode=None):
            _, C, H, W = x.shape
            x = x.reshape((rows, cols, C, H, W))
            x = x.transpose(0, 3, 1, 4, 2)

            if C == 1:
                x = x.reshape((rows*H, cols*W))

            else:
                x = x.reshape((rows*H, cols*W, C))

            preview_dir = '{}/preview'.format(dst)
            preview_path = preview_dir +\
                '/{}_{:0>8}.jpg'.format(name, trainer.updater.iteration)

            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            Image.fromarray(x, mode=mode).convert('RGB').save(preview_path)

        _gen = np.asarray(
            np.clip(cae_all * 128 + 128, 0.0, 255.0), dtype=np.uint8)
        save_image(_gen, "gen")

        _in = np.asarray(
            np.clip(in_all * 128 + 128, 0.0, 255.0), dtype=np.uint8)
        save_image(_in, "in")

    return make_image
