import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from models.PDRSI import PDRSI
from utility.dataset import MyDatasetWithBothDiscussion, get_data
from utility.matric import report_performance
from utility.utils import torch_fix_seed


def main(
    model, batch_size: int, length_technical: int, debug_test: bool, model_config: dict
):
    T_dash = 4
    prune = False
    debug_test = debug_test
    bert_type = model_config["bert_type"]
    num_epochs = model_config["num_epochs"]
    batch_size = batch_size
    lr = 0.00005
    device = torch.device("cuda", index=0) if torch.cuda.is_available() else "cpu"
    use_gpu = True

    if debug_test:
        model_config = {"alias": "debug_test_{}".format(T_dash), "bert_type": bert_type}
        num_epochs = 2

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
        train_date,
        valid_date,
        test_date,
    ) = get_data(model_config=model_config, T_dash=T_dash, prune=prune, discussion=True)

    if debug_test:
        train_text = train_text[:100, :, :]
        train_label = train_label[:100, :, :]
        train_macro = train_macro[:100, :, :]
        train_micro = train_micro[:100, :, :]
        train_date = train_date[:100]
        valid_text = train_text[:100, :, :]
        valid_label = train_label[:100, :, :]
        valid_macro = train_macro[:100, :, :]
        valid_micro = train_micro[:100, :, :]
        valid_date = valid_date[:100]
        test_text = train_text[:100, :, :]
        test_label = train_label[:100, :, :]
        test_macro = train_macro[:100, :, :]
        test_micro = train_micro[:100, :, :]
        test_date = test_date[:100]

    train_dataset = MyDatasetWithBothDiscussion(
        input_ids=train_text,
        labels=train_label,
        macro_data=train_macro,
        micro_data=train_micro,
        date=train_date,
        length_technical=length_technical,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = MyDatasetWithBothDiscussion(
        input_ids=valid_text,
        labels=valid_label,
        macro_data=valid_macro,
        micro_data=valid_micro,
        date=valid_date,
        length_technical=length_technical,
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = MyDatasetWithBothDiscussion(
        input_ids=test_text,
        labels=test_label,
        macro_data=test_macro,
        micro_data=test_micro,
        date=test_date,
        length_technical=length_technical,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(summary(model=model))

    if use_gpu:
        model.to(device)

    # loss_fct = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        epoch_train_loss = 0
        epoch_valid_loss = 0
        epoch_test_loss = 0
        model.train()
        for i, batch in tqdm(enumerate(train_dataloader)):
            input_text = batch[0]
            tweet_label = batch[1].squeeze(dim=1).float()
            history_label = batch[2].float()
            macro_data = batch[3].float()
            micro_data = batch[4].float()
            discussion = batch[5].float()
            technical_discussion = batch[6].float()
            if use_gpu:
                input_text = input_text.to(device)
                tweet_label = tweet_label.to(device)
                history_label = history_label.to(device)
                macro_data = macro_data.to(device)
                micro_data = micro_data.to(device)
                discussion = discussion.to(device)
                technical_discussion = technical_discussion.to(device)
            predict = model(
                input_text,
                tweet_label,
                history_label,
                macro_data,
                micro_data,
                discussion,
                technical_discussion,
            )
            loss = criterion(predict, tweet_label)
            epoch_train_loss += loss.detach().to("cpu").item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_predict = []
        val_label = []
        for _, batch in tqdm(enumerate(valid_dataloader)):

            input_text = batch[0]
            tweet_label = batch[1].squeeze().float()
            history_label = batch[2].float()
            macro_data = batch[3].float()
            micro_data = batch[4].float()
            discussion = batch[5].float()
            technical_discussion = batch[6].float()
            if use_gpu:
                input_text = input_text.to(device)
                tweet_label = tweet_label.to(device)
                history_label = history_label.to(device)
                macro_data = macro_data.to(device)
                micro_data = micro_data.to(device)
                discussion = discussion.to(device)
                technical_discussion = technical_discussion.to(device)
            predict = model(
                input_text,
                tweet_label,
                history_label,
                macro_data,
                micro_data,
                discussion,
                technical_discussion,
            )
            loss = criterion(predict, tweet_label)
            epoch_valid_loss += loss.detach().to("cpu").item()
            val_predict.append(predict.detach().to("cpu").numpy())
            val_label.append(tweet_label.detach().to("cpu").numpy())
        report_performance(
            np.concatenate(val_label, axis=0),
            np.concatenate(val_predict, axis=0),
            epoch_valid_loss,
            "val result",
            epoch_id=epoch,
            config=model_config,
        )

        with torch.no_grad():
            test_predict = []
            test_label = []
            for i, batch in enumerate(test_dataloader):
                input_text = batch[0]
                tweet_label = batch[1].squeeze().float()
                history_label = batch[2].float()
                macro_data = batch[3].float()
                micro_data = batch[4].float()
                discussion = batch[5].float()
                technical_discussion = batch[6].float()
                if use_gpu:
                    input_text = input_text.to(device)
                    tweet_label = tweet_label.to(device)
                    history_label = history_label.to(device)
                    macro_data = macro_data.to(device)
                    micro_data = micro_data.to(device)
                    discussion = discussion.to(device)
                    technical_discussion = technical_discussion.to(device)
                predict = model(
                    input_text,
                    tweet_label,
                    history_label,
                    macro_data,
                    micro_data,
                    discussion,
                    technical_discussion,
                )
                # loss = loss_fct(tweet_label, predict)
                loss = criterion(predict, tweet_label)
                epoch_test_loss += loss.detach().to("cpu").item()
                test_predict.append(predict.detach().to("cpu").numpy())
                test_label.append(tweet_label.detach().to("cpu").numpy())
        report_performance(
            np.concatenate(test_label, axis=0),
            np.concatenate(test_predict, axis=0),
            epoch_test_loss,
            "test result",
            epoch_id=epoch,
            config=model_config,
        )


if __name__ == "__main__":
    torch_fix_seed(seed=2020)
    batch_size = 1024
    T_dash = 4
    num_epochs = 21
    seed = 15
    debut_test = False
    technical_length_list = [7]
    bert_types = ["finbert"]
    for bert_type in bert_types:
        for technical_length in technical_length_list:
            model_config = {
                "alias": "discussion_text_and_technical_{}_timeseries_{}_{}_{}".format(
                    technical_length, T_dash, bert_type, seed
                ),
                "bert_type": bert_type,
                "num_epochs": num_epochs,
            }
            model = PDRSI(
                batch_size=batch_size,
                model_config=model_config,
                length_technical=technical_length,
            )

            main(
                model,
                batch_size=batch_size,
                length_technical=technical_length,
                debug_test=debut_test,
                model_config=model_config,
            )
