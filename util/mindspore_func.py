import mindspore
import mindspore.dataset as ds
from mindspore.dataset import vision


def get_data(batch_size: int = 128, train_percent: float = 0.8):
    data = ds.SemeionDataset(dataset_dir="./data/", shuffle=True)
    data = data.map(operations=[vision.ToType(mindspore.float32)], input_columns="image")
    data = data.map(operations=[vision.ToType(mindspore.int32)], input_columns="label")
    data = data.batch(batch_size=batch_size, drop_remainder=False)
    # 划分数据集
    train_dataset, val_dataset = data.split([train_percent, 1 - train_percent])
    return train_dataset, val_dataset
