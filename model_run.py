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


def main(model, debug_test: bool, model_config: dict):
    T_dash = model_config["T_dash"]
    prune = False
    debug_test = debug_test
    length_technical = model_config["length_technical"]
    bert_type = model_config["bert_type"]
    num_epochs = model_config["num_epochs"]
    batch_size = model_config["batch_size"]
    lr = model_config["lr"]
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
        train_date,
        valid_date,
        test_date,
    ) = get_data(model_config=model_config, T_dash=T_dash, prune=prune)

    if debug_test:
        train_text = train_text[:100, :, :]
        train_label = train_label[:100, :, :]
        train_date = train_date[:100]
        valid_text = train_text[:100, :, :]
        valid_label = train_label[:100, :, :]
        valid_date = valid_date[:100]
        test_text = train_text[:100, :, :]
        test_label = train_label[:100, :, :]
        test_date = test_date[:100]

    train_dataset = MyDatasetWithBothDiscussion(
        input_ids=train_text,
        labels=train_label,
        date=train_date,
        length_technical=length_technical,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = MyDatasetWithBothDiscussion(
        input_ids=valid_text,
        labels=valid_label,
        date=valid_date,
        length_technical=length_technical,
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = MyDatasetWithBothDiscussion(
        input_ids=test_text,
        labels=test_label,
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
            discussion = batch[3].float()
            technical_discussion = batch[4].float()
            if use_gpu:
                input_text = input_text.to(device)
                tweet_label = tweet_label.to(device)
                history_label = history_label.to(device)
                discussion = discussion.to(device)
                technical_discussion = technical_discussion.to(device)
            predict = model(
                input_text,
                history_label,
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
            discussion = batch[3].float()
            technical_discussion = batch[4].float()
            if use_gpu:
                input_text = input_text.to(device)
                tweet_label = tweet_label.to(device)
                history_label = history_label.to(device)
                discussion = discussion.to(device)
                technical_discussion = technical_discussion.to(device)
            predict = model(
                input_text,
                history_label,
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
                discussion = batch[3].float()
                technical_discussion = batch[4].float()
                if use_gpu:
                    input_text = input_text.to(device)
                    tweet_label = tweet_label.to(device)
                    history_label = history_label.to(device)
                    discussion = discussion.to(device)
                    technical_discussion = technical_discussion.to(device)
                predict = model(
                    input_text,
                    history_label,
                    discussion,
                    technical_discussion,
                )
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
    SEED = 2020
    BATCH_SIZE = 1024
    T_DASH = 4
    NUM_EPOCHS = 21
    debut_test = True
    LENGTH_TECHNICAL = 7
    LEARNING_RATE = 0.0005
    BERT_TYPE = "finbert"
    model_config = {
        "alias": "PDRSI_",
        "bert_type": BERT_TYPE,
        "num_epochs": NUM_EPOCHS,
        "T_dash": T_DASH,
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "length_technical": LENGTH_TECHNICAL,
    }
    torch_fix_seed(seed=SEED)
    model = PDRSI(model_config=model_config)

    main(
        model,
        debug_test=debut_test,
        model_config=model_config,
    )
