import os
import datetime

from util.write_text import WriteText


def environment_log(args, save_path):
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = save_path / now
    save_dir.mkdir(exist_ok=True, parents=True)

    environment = WriteText()
    environment.add('# datetime: {}'.format(now))
    environment.add('# using machine: [{}]'.format(os.uname()[1]))
    environment.add('# training dataset: {}'.format(args.dataset))
    environment.add('# GPU: {}'.format(args.gpu))
    environment.add('# Minibatch-size: {}'.format(args.batchsize))
    environment.add('# iteration: {}'.format(args.iteration))
    environment.add('# optimizer: {}'.format(args.optimizer))
    environment.add('')
    environment.save(str(save_dir / 'environment.txt'))

    return save_dir
