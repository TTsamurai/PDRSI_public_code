import pickle
import random
from collections import defaultdict
from datetime import date
from datetime import datetime as dt
import numpy as np
from typing import List
import pandas as pd
import torch
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer


def turn_data_with_date_and_set_index(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = (
        df["created_at"]
        .apply(lambda x: x[:10])
        .apply(lambda x: dt.strptime(x, "%Y-%m-%d"))
        .apply(lambda x: date(x.year, x.month, x.day))
    )
    return df


def create_basic_data(data_path: str) -> pd.DataFrame:
    data = (
        pd.read_csv(data_path + "sample_data.csv")
        .sort_values(["author.id", "created_at"], ascending=False)
        .reset_index(drop=True)
    )
    text_data = data.text.to_list()
    ids = data["author.id"].map(str).to_list()
    stock_unique = np.sort(data.ticker.unique())
    stock_name_to_label_dict = dict(zip(stock_unique, list(range(len(stock_unique)))))
    label = data.ticker.map(stock_name_to_label_dict).apply(lambda x: [x]).to_list()
    one_hot = np.identity(502)[data.ticker.map(stock_name_to_label_dict).to_list()]
    data = turn_data_with_date_and_set_index(df=data)
    date_timestamp = data.Date.to_list()
    with open(data_path + "texts.pkl", "wb") as f:
        pickle.dump(text_data, f)

    with open(data_path + "labels.pkl", "wb") as f:
        pickle.dump(label, f)

    with open(data_path + "ids.pkl", "wb") as f:
        pickle.dump(ids, f)

    with open(data_path + "date.pkl", "wb") as f:
        pickle.dump(date_timestamp, f)

    with open(data_path + "one_hot_labels.pkl", "wb") as f:
        pickle.dump(one_hot, f)

    return None


def tokenize_text_data(texts: list, model_name: str) -> list:
    assert model_name in ["bert", "roberta", "bertweet", "finbert"]
    if model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif model_name == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    elif model_name == "bertweet":
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
    elif model_name == "finbert":
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    else:
        raise ValueError("Choose correct bert type")
    texts_tokenized = list(
        map(
            lambda examples: tokenizer(
                examples,
                padding="max_length",
                max_length=128,
                truncation=True,
                return_tensors="pt",
            )["input_ids"],
            texts,
        )
    )
    return texts_tokenized


def build_history_list_by_id_dic(id_dic: dict, T_dash: int):
    history_list: List[list] = []
    for _, index_list in id_dic.items():
        for index in index_list:
            history: List[int] = []
            for history_candidate in index_list:
                if history_candidate >= index and len(history) < T_dash + 2:
                    history.append(history_candidate)
            if len(history) == T_dash + 2:
                history_list.append(history)
    return np.asarray(history_list, dtype=np.int32)


def create_test_id_dict(id_dic: dict, T_dash: int) -> dict:
    return {author_id: index[: T_dash + 2] for author_id, index in id_dic.items()}


def create_valid_id_dict(id_dic: dict, T_dash: int) -> dict:
    return {
        author_id: index[random.choice(list(range(1, len(index[: -T_dash - 1])))) :][
            : T_dash + 2
        ]
        for author_id, index in id_dic.items()
    }


def create_train_id_dict(id_dic: dict, test_id_dict: dict, valid_id_dict: dict) -> dict:
    test_t_index = [test_sequence[0] for test_sequence in list(test_id_dict.values())]
    valid_t_index = [
        valid_sequence[0] for valid_sequence in list(valid_id_dict.values())
    ]
    set_ticker_exclude_from_train_data = set(test_t_index) | set(valid_t_index)
    return {
        author_id: [
            ind for ind in index if ind not in set_ticker_exclude_from_train_data
        ]
        for author_id, index in id_dic.items()
    }


def build_history(ids: list, T_dash: int):
    id_dic = defaultdict(list)
    # 空の文字列がないから0からにする
    for i in range(0, len(ids)):
        id_dic[ids[i]].append(i)
    test_id_dict = create_test_id_dict(id_dic=id_dic, T_dash=T_dash)
    valid_id_dict = create_valid_id_dict(id_dic=id_dic, T_dash=T_dash)
    train_id_dict = create_train_id_dict(
        id_dic=id_dic, test_id_dict=test_id_dict, valid_id_dict=valid_id_dict
    )
    assert set(test_id_dict.keys()) == set(
        valid_id_dict.keys()
    ), "Some samples are lost during valid set"
    #! ここで両方test_id_dictになっているのは誤り
    assert set(test_id_dict.keys()) == set(
        train_id_dict.keys()
    ), "Some samples are lost when creating training samples and test samples"

    return (
        build_history_list_by_id_dic(id_dic=train_id_dict, T_dash=T_dash),
        build_history_list_by_id_dic(id_dic=valid_id_dict, T_dash=T_dash),
        build_history_list_by_id_dic(id_dic=test_id_dict, T_dash=T_dash),
    )


def build_test_history(ids: list, T_dash: int):
    id_dic = defaultdict(list)
    # 空の文字列がないから0からにする
    for i in range(0, len(ids)):
        id_dic[ids[i]].append(i)
    test_id_dict = create_test_id_dict(id_dic=id_dic, T_dash=T_dash)

    return build_history_list_by_id_dic(id_dic=test_id_dict, T_dash=T_dash)


def build_tweet_text_data(
    hist_id_List: np.ndarray, tokenized_texts: list, T_dash: int
) -> np.ndarray:
    tweet_numpy = torch.cat(tokenized_texts, dim=0).numpy()
    his_index = hist_id_List.flatten()
    his = tweet_numpy[his_index, :]
    return np.reshape(his, (-1, T_dash + 2, his.shape[1]))


def build_cashtag_label_data(
    hist_id_List: np.ndarray, onehot: np.ndarray, T_dash: int
) -> np.ndarray:
    his_index = hist_id_List.flatten()
    his = onehot[his_index, :]
    return np.reshape(his, (-1, T_dash + 2, his.shape[1]))


def build_timestamp_data(
    hist_id_List: np.ndarray, time_stamp: list, T_dash: int
) -> np.ndarray:
    time_stamp_arr = np.array(time_stamp)
    his_index = hist_id_List.flatten()
    his = time_stamp_arr[his_index, :]
    time_stamp_reshaped = np.reshape(his, (-1, T_dash + 2, his.shape[1])).squeeze()
    return time_stamp_reshaped


def get_text_and_label_data(
    model_config: dict, data_path: str, split: bool, T_dash: int, prune: bool = False #! Add split argument
):
    #! バイナリファイルなら型付きのほうが親切だと思う
    with open(data_path + "texts.pkl", "rb") as f:
        texts = pickle.load(f)

    with open(data_path + "ids.pkl", "rb") as f:
        ids = pickle.load(f)

    with open(data_path + "one_hot_labels.pkl", "rb") as f:
        one_hot = pickle.load(f)

    with open(data_path + "date.pkl", "rb") as f:
        date_timestamp = pickle.load(f)

    if prune:
        texts = texts[:10000]
        ids = ids[:10000]
        one_hot = one_hot[:10000]
        date_timestamp = date_timestamp[:10000]

    train_hist_id_List, valid_hist_id_List, test_hist_id_List = build_history(
        ids, T_dash=T_dash
    )
    tokenized_text = tokenize_text_data(
        texts=texts, model_name=model_config["bert_type"]
    )
    #! splitごとに異なる
    if split == "train":
        hist_id_List = train_hist_id_List
    elif split == "valid":
        hist_id_List = valid_hist_id_List
    elif split == "test":
        hist_id_List = test_hist_id_List
    else:
        raise ValueError(f"Invalid split: {split}. It must be train, valid, or test.")
    text = build_tweet_text_data(
        hist_id_List=hist_id_List, tokenized_texts=tokenized_text, T_dash=T_dash
    )
    
    label = build_cashtag_label_data(
        hist_id_List=hist_id_List, onehot=one_hot, T_dash=T_dash
    )

    #! 読み込み，build_historyを重複して行うから冗長なget_date_dataは不要
    data_date = build_timestamp_data(
        hist_id_List=hist_id_List, time_stamp=date_timestamp, T_dash=T_dash
    )

    return (text, label, data_date)
