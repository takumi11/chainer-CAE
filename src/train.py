#! /usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
import cv2
from pathlib import Path

from util.args import parser
from util.get_dataset import get_datasets
from util.environment_log import environment_log

from model.cae import DeepCAE

import chainer
from chainer import training
from chainer.training import extensions
from lib.extensions.cosine_shift import CosineShift
from lib.extensions.visualizer import out_image
from lib.updater.updater import Updater


matplotlib.use('Agg')
cv2.setNumThreads(0)


def make_optimizer_adam(model, alpha=0.0002, beta1=0.5):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
    return optimizer


def make_optimizer_msgd(model, lr=0.001, momentum=0.9):
    optimizer = chainer.optimizers.MomentumSGD(lr=lr, momentum=momentum)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')

    return optimizer


def main():

    args = parser()
    cae = DeepCAE()
    save_path = Path('../result')
    save_dir = environment_log(args, save_path)

    input_data = Path(args.dataset)
    train, test, val, image_ch, target_ch = get_datasets(input_data)
    print("finish load datasets !")
    print('')

    if args.optimizer == 'adam':
        opt_cae = make_optimizer_adam(cae)
        param = 'alpha'
    else:
        opt_cae = make_optimizer_msgd(cae)
        param = 'lr'

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize)

    updater = Updater(
        model=cae,
        iterator={
            'main': train_iter,
            'test': test_iter},
        optimizer={'cae': opt_cae},
        loss_function=args.loss_function,
        device=args.gpu)

    trainer = training.Trainer(
        updater, (args.iteration, 'iteration'), out=str(save_dir))

    display_interval = (100, 'iteration')
    snapshot_interval = (args.snapshot_interval, 'iteration')

    if args.cos_shift:
        trainer.extend(CosineShift(param, args.epoch, 1, optimizer=opt_cae),
                       trigger=(10, "iteration"))
    trainer.extend(extensions.snapshot_object(cae, 'snap_model.npz'),
                   trigger=chainer.training.triggers.MinValueTrigger(
                   key='cae/loss', trigger=display_interval))
    trainer.extend(extensions.LogReport(log_name='log.json',
                                        trigger=display_interval))
    print_rep = ['epoch', 'iteration', 'cae/loss', 'elapsed_time']
    trainer.extend(extensions.PrintReport(print_rep), display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(out_image(updater, cae, image_ch, target_ch,
                             5, 5, args.seed, str(save_dir)),
                   trigger=snapshot_interval)

    trainer.run()


if __name__ == '__main__':
    print('\nstart program ...')
    main()
    print('finish .\n')
