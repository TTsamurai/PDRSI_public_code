from datetime import date
from datetime import datetime as dt

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer
from model_components.bert import Bert


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


def turn_data_with_date(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = (
        df["created_at"]
        .apply(lambda x: x[:10])
        .apply(lambda x: dt.strptime(x, "%Y-%m-%d"))
        .apply(lambda x: date(x.year, x.month, x.day))
    )
    return df


def calculate_bert_model_with_gpu(
    text: torch.Tensor, model, device=torch.device("cuda", index=0)
) -> list:
    model.to(device)
    text = text.to(device)
    output = model(text)
    return output.to("cpu").detach().numpy()


def create_hot_discussion(data_path: str) -> pd.DataFrame:
    filtered_data_path = data_path + "sample_data.csv"
    df = pd.read_csv(filtered_data_path)
    df_with_date = turn_data_with_date(df)
    data_unique = df_with_date["Date"].unique()
    ticker_unique = df_with_date["ticker"].unique()
    model_config = {"bert_type": "finbert"}
    bert_model = Bert(model_config=model_config)
    full_zeros_data = pd.DataFrame(
        np.zeros((len(data_unique) * len(ticker_unique), 1)),
        index=pd.MultiIndex.from_product(
            [data_unique, ticker_unique], names=["Date", "ticker"]
        ),
    )
    df_with_date_text_in_a_list = pd.DataFrame(
        df_with_date.groupby(["Date", "ticker"])["text"].agg(list)
    )
    data_with_text_full_index = pd.merge(
        full_zeros_data,
        df_with_date_text_in_a_list,
        right_index=True,
        left_index=True,
        how="left",
    )[["text"]]
    data_with_text_full_index_space_padding = data_with_text_full_index.fillna(" ")
    text_tokenized = data_with_text_full_index_space_padding.apply(
        lambda x: tokenize_text_data(x, model_name="finbert")
    )
    embed = []
    for date_val in tqdm(np.sort(data_unique)):
        chunked_data = text_tokenized[
            text_tokenized.index.get_level_values(0) == date_val
        ]
        check = chunked_data.applymap(
            lambda x: calculate_bert_model_with_gpu(x, model=bert_model)
        ).apply(lambda x: x[0].mean(axis=0), axis=1)

        embed.append(pd.DataFrame(np.vstack(check.values), index=check.index))
    pd.concat(embed, axis=0).to_csv(data_path + "hot_discussion.csv")


if __name__ == "__main__":
    data_path = "./data/"
    create_hot_discussion(data_path=data_path)
