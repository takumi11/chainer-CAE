import os
from PIL import Image
import numpy as np
from tqdm import tqdm

from chainer.dataset import dataset_mixin


class LoadDataset(dataset_mixin.DatasetMixin):
    def __init__(self, inputDir):

        self.imgDir = inputDir

        self.dataset = []
        dir_name = os.listdir(self.imgDir)
        mean = 'mean.npy'
        if mean in dir_name:
            dir_name.remove(mean)

        print("load dataset start")

        for di in dir_name:
            print("now loading {} directory".format(di))
            img_dir = inputDir / di
            files = os.listdir(str(img_dir))

            for file_name in tqdm(files):
                img = Image.open(str(img_dir / file_name))
                img = np.asarray(img).astype("f").transpose(2, 0, 1)
                label = np.int32(file_name[0])
                self.dataset.append((img, img, label))

        print("load dataset done")
        self.input_ch = img.shape[0]
        self.output_ch = img.shape[0]
        print('image_ch: {}'.format(self.input_ch))
        print('output_ch: {}'.format(self.output_ch))

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):

        _input = self.dataset[i][0]
        _output = self.dataset[i][1]

        return _input, _output
