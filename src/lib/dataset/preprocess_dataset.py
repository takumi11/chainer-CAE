from chainer.dataset import dataset_mixin


class PreprocessDataset(dataset_mixin.DatasetMixin):
    def __init__(self, pair, transform=False):
        self.base = pair

    def __len__(self):
        return len(self.base)

    def get_example(self, i, crop_size=256):
        img, label = self.base[i]
        _, h, w = img.shape

        img = img / 128.0 - 1.0
        label = label / 128.0 - 1.0

        return img, label
