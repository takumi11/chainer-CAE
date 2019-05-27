import argparse


def parser():
    parser = argparse.ArgumentParser(
        description='chainer implementation of pix2pix')
    parser.add_argument('--batchsize', '-b', type=int, default=8,
                        help='Number of images in each mini-batch')

    parser.add_argument('--iteration', '-i', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')

    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')

    parser.add_argument('--dataset', default='',
                        help='input dataset')

    parser.add_argument('--optimizer', default='adam',
                        help='using discriminator optimizer')

    parser.add_argument('--loss_function', default='mse',
                        help='using loss function')

    parser.add_argument('--cos_shift',
                        action='store_true',
                        help="use or not use cos_shift")

    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')

    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    parser.add_argument('--snapshot_interval', type=int, default=100,
                        help='Interval of snapshot')

    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')

    args = parser.parse_args()

    return args
