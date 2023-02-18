import os
from datetime import date
from datetime import datetime as dt

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

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
        macro_data,
        micro_data,
        date,
        length_technical,
        data_path: str = "./data/",
    ):
        super().__init__()
        self.input_ids = input_ids[:, 1:, :]
        self.tweet_labels = labels[:, :1, :]
        self.history_labels = labels[:, 1:, :]
        self.macro_data_at_t_minus_1 = macro_data[:, 1, :]
        self.micro_input_data = micro_data[:, 1:, :]
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
            self.tweet_labels[index],
            self.history_labels[index],
            self.macro_data_at_t_minus_1[index],
            self.micro_input_data[index],
            self.take_discussion_data(
                discussion=self.discussion, date_index=self.date[index]
            ),
            self.take_discussion_data(
                discussion=self.technical_discussion, date_index=self.date[index]
            ),
        )


def get_data(
    model_config: dict,
    T_dash: int,
    prune: bool = False,
    project_path: str = "./",
    discussion: bool = False,
):
    data_path = project_path + "data/"
    bert_type = model_config["bert_type"]
    data_train_text_path = project_path + "data/train/text_{}_{}.npy".format(
        T_dash, bert_type
    )
    data_valid_text_path = project_path + "data/valid/text_{}_{}.npy".format(
        T_dash, bert_type
    )
    data_test_text_path = project_path + "data/test/text_{}_{}.npy".format(
        T_dash, bert_type
    )
    data_train_label_path = project_path + "data/train/label_{}.npy".format(T_dash)
    data_valid_label_path = project_path + "data/valid/label_{}.npy".format(T_dash)
    data_test_label_path = project_path + "data/test/label_{}.npy".format(T_dash)
    data_train_macro_path = project_path + "data/train/macro_{}.npy".format(T_dash)
    data_valid_macro_path = project_path + "data/valid/macro_{}.npy".format(T_dash)
    data_test_macro_path = project_path + "data/test/macro_{}.npy".format(T_dash)
    data_train_micro_path = project_path + "data/train/micro_{}.npy".format(T_dash)
    data_valid_micro_path = project_path + "data/valid/micro_{}.npy".format(T_dash)
    data_test_micro_path = project_path + "data/test/micro_{}.npy".format(T_dash)
    if discussion:
        data_train_date_path = "./data/train/date_{}.npy".format(T_dash)
        data_valid_date_path = "./data/valid/date_{}.npy".format(T_dash)
        data_test_date_path = "./data/test/date_{}.npy".format(T_dash)

    if os.path.exists(data_train_text_path):
        train_text = np.load(data_train_text_path)
        valid_text = np.load(data_valid_text_path)
        test_text = np.load(data_test_text_path)
        train_label = np.load(data_train_label_path)
        valid_label = np.load(data_valid_label_path)
        test_label = np.load(data_test_label_path)
        train_macro = np.load(data_train_macro_path)
        valid_macro = np.load(data_valid_macro_path)
        test_macro = np.load(data_test_macro_path)
        train_micro = np.load(data_train_micro_path)
        valid_micro = np.load(data_valid_micro_path)
        test_micro = np.load(data_test_micro_path)
        if discussion:
            train_date = np.load(data_train_date_path, allow_pickle=True)
            valid_date = np.load(data_valid_date_path, allow_pickle=True)
            test_date = np.load(data_test_date_path, allow_pickle=True)

    else:
        (
            train_text,
            valid_text,
            test_text,
            train_label,
            valid_label,
            test_label,
            train_macro,
            valid_macro,
            test_macro,
            train_micro,
            valid_micro,
            test_micro,
        ) = get_text_and_label_data(
            model_config=model_config, data_path=data_path, T_dash=T_dash, prune=prune
        )
        np.save(data_train_text_path, train_text)
        np.save(data_valid_text_path, valid_text)
        np.save(data_test_text_path, test_text)
        np.save(data_train_label_path, train_label)
        np.save(data_valid_label_path, valid_label)
        np.save(data_test_label_path, test_label)
        np.save(data_train_macro_path, train_macro)
        np.save(data_valid_macro_path, valid_macro)
        np.save(data_test_macro_path, test_macro)
        np.save(data_train_micro_path, train_micro)
        np.save(data_valid_micro_path, valid_micro)
        np.save(data_test_micro_path, test_micro)

    if discussion:
        return (
            train_text,
            valid_text,
            test_text,
            train_label,
            valid_label,
            test_label,
            train_macro,
            valid_macro,
            test_macro,
            train_micro,
            valid_micro,
            test_micro,
            train_date,
            valid_date,
            test_date,
        )

    return (
        train_text,
        valid_text,
        test_text,
        train_label,
        valid_label,
        test_label,
        train_macro,
        valid_macro,
        test_macro,
        train_micro,
        valid_micro,
        test_micro,
    )


if __name__ == "__main__":
    model_config = {"bert_type": "bert"}
    (
        train_text,
        valid_text,
        test_text,
        train_label,
        valid_label,
        test_label,
        train_macro,
        valid_macro,
        test_macro,
        train_micro,
        valid_micro,
        test_micro,
    ) = get_data(model_config=model_config, T_dash=4, project_path="../")


class MyFrequentDataset(Dataset):
    """
    Return:
    Tweet Label (16, 1, 50)
    History label (16, 5, 50)
    """

    def __init__(self, input_ids, labels, macro_data, micro_data):
        super().__init__()
        # tweet at time t　は使わない
        self.input_ids = input_ids[:, 1:, :]
        # tweet at time t　をラベルに
        self.tweet_labels = labels[:, :1, :]
        # tweet at time t-2 ~ t-T-1をInputとして用いる
        self.history_labels = labels[:, 1:, :]
        # time t-1のmacroデータをconcatする　(time tの予測をするのに対して、ここはt-1 ~ t-Tのどの長さでも構わない）
        self.macro_data_at_t_minus_1 = macro_data[:, 1, :]
        self.micro_input_data = micro_data[:, 1:, :]
        self.len = len(input_ids)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (
            self.input_ids[index],
            self.tweet_labels[index],
            self.history_labels[index],
            self.macro_data_at_t_minus_1[index],
            self.micro_input_data[index],
        )
