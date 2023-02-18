import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer


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
    history_list = []
    for _, index_list in id_dic.items():
        for index in index_list:
            history = []
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
    assert set(test_id_dict.keys()) == set(
        test_id_dict.keys()
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
    tweet_numpy = torch.cat(tokenized_texts, axis=0).numpy()
    his_index = hist_id_List.flatten()
    his = tweet_numpy[his_index, :]
    return np.reshape(his, (-1, T_dash + 2, his.shape[1]))


def build_cashtag_label_data(
    hist_id_List: np.ndarray, onehot: np.ndarray, T_dash: int
) -> np.ndarray:
    his_index = hist_id_List.flatten()
    his = onehot[his_index, :]
    return np.reshape(his, (-1, T_dash + 2, his.shape[1]))


def build_macro_data(
    hist_id_List: np.ndarray, macro_data: list, T_dash: int
) -> np.ndarray:
    macro_data_numpy = np.vstack(macro_data)
    his_index = hist_id_List.flatten()
    his = macro_data_numpy[his_index, :]
    return np.reshape(his, (-1, T_dash + 2, his.shape[1]))


def build_micro_data(
    hist_id_List: np.ndarray, micro_data: list, T_dash: int
) -> np.ndarray:
    macro_data_numpy = np.vstack(micro_data)
    his_index = hist_id_List.flatten()
    his = macro_data_numpy[his_index, :]
    return np.reshape(his, (-1, T_dash + 2, his.shape[1]))


def build_timestamp_data(
    hist_id_List: np.ndarray, time_stamp: list, T_dash: int
) -> np.ndarray:
    time_stamp = np.array(time_stamp)
    time_stamp = time_stamp.reshape(-1, 1)
    his_index = hist_id_List.flatten()
    his = time_stamp[his_index, :]
    # t-1期のTimeStampのみにする？この作業は今後というかあとででよいか
    time_stamp = np.reshape(his, (-1, T_dash + 2, his.shape[1])).squeeze()
    return time_stamp


def get_text_and_label_data(
    model_config: dict, data_path: str, T_dash: int, prune: bool = False
):
    with open(data_path + "texts.pkl", "rb") as f:
        texts = pickle.load(f)

    with open(data_path + "ids.pkl", "rb") as f:
        ids = pickle.load(f)

    with open(data_path + "one_hot_labels.pkl", "rb") as f:
        one_hot = pickle.load(f)

    with open(data_path + "macro_technical.pkl", "rb") as f:
        macro_technical = pickle.load(f)

    with open(data_path + "micro_technical.pkl", "rb") as f:
        micro_technical = pickle.load(f)

    with open(data_path + "date.pkl", "rb") as f:
        date_timestamp = pickle.load(f)

    if prune:
        texts = texts[:10000]
        ids = ids[:10000]
        one_hot = one_hot[:10000]
        macro_technical = macro_technical[:10000]
        micro_technical = micro_technical[:10000]
        date_timestamp = date_timestamp[:10000]

    train_hist_id_List, valid_hist_id_List, test_hist_id_List = build_history(
        ids, T_dash=T_dash
    )
    tokenized_text = tokenize_text_data(
        texts=texts, model_name=model_config["bert_type"]
    )

    train_text = build_tweet_text_data(
        hist_id_List=train_hist_id_List, tokenized_texts=tokenized_text, T_dash=T_dash
    )
    valid_text = build_tweet_text_data(
        hist_id_List=valid_hist_id_List, tokenized_texts=tokenized_text, T_dash=T_dash
    )
    test_text = build_tweet_text_data(
        hist_id_List=test_hist_id_List, tokenized_texts=tokenized_text, T_dash=T_dash
    )

    train_label = build_cashtag_label_data(
        hist_id_List=train_hist_id_List, onehot=one_hot, T_dash=T_dash
    )

    valid_label = build_cashtag_label_data(
        hist_id_List=valid_hist_id_List, onehot=one_hot, T_dash=T_dash
    )

    test_label = build_cashtag_label_data(
        hist_id_List=test_hist_id_List, onehot=one_hot, T_dash=T_dash
    )

    train_macro = build_macro_data(
        hist_id_List=train_hist_id_List, macro_data=macro_technical, T_dash=T_dash
    )
    valid_macro = build_macro_data(
        hist_id_List=valid_hist_id_List, macro_data=macro_technical, T_dash=T_dash
    )
    test_macro = build_macro_data(
        hist_id_List=valid_hist_id_List, macro_data=macro_technical, T_dash=T_dash
    )

    train_micro = build_micro_data(
        hist_id_List=train_hist_id_List, micro_data=micro_technical, T_dash=T_dash
    )
    valid_micro = build_micro_data(
        hist_id_List=valid_hist_id_List, micro_data=micro_technical, T_dash=T_dash
    )
    test_micro = build_micro_data(
        hist_id_List=valid_hist_id_List, micro_data=micro_technical, T_dash=T_dash
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


def get_text_label_data_with_T_dash(data_path: str, T_dash: int, prune: bool = False):
    with open(data_path + "texts.pkl", "rb") as f:
        texts = pickle.load(f)

    with open(data_path + "ids.pkl", "rb") as f:
        ids = pickle.load(f)

    with open(data_path + "one_hot_labels.pkl", "rb") as f:
        one_hot = pickle.load(f)

    with open(data_path + "macro_technical.pkl", "rb") as f:
        macro_technical = pickle.load(f)

    with open(data_path + "micro_technical.pkl", "rb") as f:
        micro_technical = pickle.load(f)

    with open(data_path + "date.pkl", "rb") as f:
        date_timestamp = pickle.load(f)

    if prune:
        texts = texts[:10000]
        ids = ids[:10000]
        one_hot = one_hot[:10000]
        macro_technical = macro_technical[:10000]
        micro_technical = micro_technical[:10000]
        date_timestamp = date_timestamp[:10000]

    test_hist_id_List = build_test_history(ids, T_dash=T_dash)

    test_label = build_cashtag_label_data(
        hist_id_List=test_hist_id_List, onehot=one_hot, T_dash=T_dash
    )

    return test_label


def get_date_data(data_path: str, T_dash: int, prune: bool = False):
    with open(data_path + "ids.pkl", "rb") as f:
        ids = pickle.load(f)

    with open(data_path + "one_hot_labels.pkl", "rb") as f:
        one_hot = pickle.load(f)

    with open(data_path + "date.pkl", "rb") as f:
        date_timestamp = pickle.load(f)

    if prune:
        ids = ids[:10000]
        one_hot = one_hot[:10000]
        date_timestamp = date_timestamp[:10000]

    train_hist_id_List, valid_hist_id_List, test_hist_id_List = build_history(
        ids, T_dash=T_dash
    )

    train_date = build_timestamp_data(
        hist_id_List=train_hist_id_List, time_stamp=date_timestamp, T_dash=T_dash
    )

    valid_date = build_timestamp_data(
        hist_id_List=valid_hist_id_List, time_stamp=date_timestamp, T_dash=T_dash
    )

    test_date = build_timestamp_data(
        hist_id_List=test_hist_id_List, time_stamp=date_timestamp, T_dash=T_dash
    )

    return (train_date, valid_date, test_date)


def create_text_data_for_different_bert_model(
    model_name: str,
    data_path: str,
    T_dash: int,
    prune: bool = False,
    save_text: bool = False,
):
    with open(data_path + "texts.pkl", "rb") as f:
        texts = pickle.load(f)

    with open(data_path + "ids.pkl", "rb") as f:
        ids = pickle.load(f)

    with open(data_path + "one_hot_labels.pkl", "rb") as f:
        one_hot = pickle.load(f)

    with open(data_path + "macro_technical.pkl", "rb") as f:
        macro_technical = pickle.load(f)

    with open(data_path + "micro_technical.pkl", "rb") as f:
        micro_technical = pickle.load(f)

    with open(data_path + "date.pkl", "rb") as f:
        date_timestamp = pickle.load(f)
    if prune:
        texts = texts[:10000]
        ids = ids[:10000]
        one_hot = one_hot[:10000]
        macro_technical = macro_technical[:10000]
        micro_technical = micro_technical[:10000]
        date_timestamp = date_timestamp[:10000]

    train_hist_id_List, valid_hist_id_List, test_hist_id_List = build_history(
        ids, T_dash=T_dash
    )
    tokenized_text = tokenize_text_data(texts=texts, model_name=model_name)

    train_text = build_tweet_text_data(
        hist_id_List=train_hist_id_List, tokenized_texts=tokenized_text, T_dash=T_dash
    )
    valid_text = build_tweet_text_data(
        hist_id_List=valid_hist_id_List, tokenized_texts=tokenized_text, T_dash=T_dash
    )
    test_text = build_tweet_text_data(
        hist_id_List=test_hist_id_List, tokenized_texts=tokenized_text, T_dash=T_dash
    )

    train_date = build_timestamp_data(
        hist_id_List=train_hist_id_List, time_stamp=date_timestamp, T_dash=T_dash
    )

    valid_date = build_timestamp_data(
        hist_id_List=valid_hist_id_List, time_stamp=date_timestamp, T_dash=T_dash
    )

    test_date = build_timestamp_data(
        hist_id_List=test_hist_id_List, time_stamp=date_timestamp, T_dash=T_dash
    )

    train_text_save_path = data_path + "train/text_{}_{}".format(T_dash, model_name)
    valid_text_save_path = data_path + "valid/text_{}_{}".format(T_dash, model_name)
    test_text_save_path = data_path + "test/text_{}_{}".format(T_dash, model_name)
    if save_text:
        np.save(train_text_save_path, train_text)
        np.save(valid_text_save_path, valid_text)
        np.save(test_text_save_path, test_text)


if __name__ == "__main__":
    T_dash = 4
    data_path = "./data/"
    model_config = {"bert_type": "bert"}
    test_id = get_text_label_data_with_T_dash(data_path=data_path, T_dash=T_dash)
