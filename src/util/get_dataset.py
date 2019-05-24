import chainer
from lib.dataset.load_dataset import LoadDataset
from lib.dataset.preprocess_dataset import PreprocessDataset


def get_datasets(input_data, split=0.9):

    dataset = LoadDataset(inputDir=input_data)
    in_ch = dataset.input_ch
    out_ch = dataset.output_ch

    split_at = int(len(dataset) * split)
    train_dataset, test_dataset = chainer.datasets.split_dataset_random(
        dataset, split_at, seed=0)
    train = PreprocessDataset(train_dataset)
    test = PreprocessDataset(test_dataset)
    val = PreprocessDataset(test_dataset)

    print('train length : {}'.format(len(train)))
    print('test length : {}'.format(len(test)))

    return train, test, val, in_ch, out_ch
