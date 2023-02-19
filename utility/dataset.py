import os
from datetime import date
from datetime import datetime as dt

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from preprocess.create_data import get_text_and_label_data
from preprocess.create_technical_discussion import get_n_days_technical_discussion


class MyDatasetWithBothDiscussion(Dataset):
    """
    Return:
    Input_text (16, 5, 128)
    Tweet Label (16, 1, 50)
    History label (16, 5, 50)
    """

    def __init__(
        self,
        input_ids,
        labels,
        date,
        length_technical,
        data_path: str = "./data/",
    ):
        super().__init__()
        self.input_ids = input_ids[:, 1:, :]
        self.tweet_labels = labels[:, :1, :]
        self.history_labels = labels[:, 1:, :]
        self.date = date[:, 1]
        self.discussion = self.load_discussion_data(data_path=data_path)
        self.technical_discussion = self.load_technical_discussion_data(
            data_path=data_path, n=length_technical
        )

        self.len = len(input_ids)

    def __len__(self):
        return self.len

    def take_discussion_data(
        self, discussion: pd.DataFrame, date_index: np.ndarray
    ) -> np.ndarray:
        return discussion[discussion.Date == date_index].drop("Date", axis=1).values

    def load_technical_discussion_data(
        self, n: int, data_path: str, normalize: bool = True
    ) -> pd.DataFrame:
        technical_discussion = get_n_days_technical_discussion(
            n=n, project_data_path=data_path
        ).drop("ticker", axis=1)
        if normalize:
            technical_discussion = (
                technical_discussion.set_index("Date")
                .apply(lambda x: (x - x.mean()) / x.std())
                .reset_index()
            )

        return technical_discussion

    def load_discussion_data(self, data_path: str) -> pd.DataFrame:
        discussion = pd.read_csv(data_path + "hot_discussion.csv").drop(
            "ticker", axis=1
        )
        discussion["Date"] = discussion.Date.apply(
            lambda x: dt.strptime(x, "%Y-%m-%d")
        ).apply(lambda x: date(x.year, x.month, x.day))
        return discussion

    def __getitem__(self, index):
        return (
            self.input_ids[index],
            self.history_labels[index],
            self.take_discussion_data(
                discussion=self.discussion, date_index=self.date[index]
            ),
            self.take_discussion_data(
                discussion=self.technical_discussion, date_index=self.date[index]
            ),
            self.tweet_labels[index],
        )


def get_dataset_data(
    model_config: dict,
    T_dash: int,
    split: str,
    prune: bool = False,
    project_path: str = "./",
):
    assert split in {
        "train",
        "valid",
        "test",
    }, f"Invalid split: {split}. It must be train, valid, or test."
    data_path = project_path + "data/"
    bert_type = model_config["bert_type"]
    data_text_path = project_path + "data/{}/text_{}_{}.npy".format(
        split, T_dash, bert_type
    )
    data_label_path = project_path + "data/{}/label_{}.npy".format(split, T_dash)
    data_date_path = "./data/{}/date_{}.npy".format(split, T_dash)

    if os.path.exists(data_text_path):
        data_text = np.load(data_text_path)
        data_label = np.load(data_label_path)
        data_date = np.load(data_date_path, allow_pickle=True)

    else:
        data_text, data_label, data_date = get_text_and_label_data(
            model_config=model_config,
            data_path=data_path,
            split=split,
            T_dash=T_dash,
            prune=prune,
        )

        np.save(data_text_path, data_text)
        np.save(data_label_path, data_label)
        np.save(data_date_path, data_date)

    return (data_text, data_label, data_date)


def get_dataloader(
    model_config: dict,
    T_dash: int,
    split: str,
    length_technical: int,
    batch_size: int,
    prune: bool = False,
    debug_test: bool = False,
) -> DataLoader:
    assert split in {
        "train",
        "valid",
        "test",
    }, f"Invalid split: {split}. It must be train, valid, or test."
    data_text, data_label, data_date = get_dataset_data(
        model_config=model_config, split=split, T_dash=T_dash, prune=prune
    )
    if debug_test:
        data_text = data_text[:100, :, :]
        data_label = data_label[:100, :, :]
        data_date = data_date[:100]
    dataset = MyDatasetWithBothDiscussion(
        input_ids=data_text,
        labels=data_label,
        date=data_date,
        length_technical=length_technical,
    )
    shuffle: bool = bool(split == "train")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
