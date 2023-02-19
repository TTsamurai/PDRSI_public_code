import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torchinfo import summary
from tqdm import tqdm
from typing import Dict, List
from torch.utils.data import DataLoader
from models.PDRSI import PDRSI
from utility.dataset import (
    get_dataloader,
)  
from utility.matric import report_performance
from utility.utils import torch_fix_seed


def move_to_cuda(
    batch: List[torch.Tensor], model: nn.Module, device: torch.device
) -> Dict[str, torch.Tensor]:
    model.to(device=device)
    input_text = batch[0]
    history_label = batch[1].float()
    discussion = batch[2].float()
    technical_discussion = batch[3].float()
    tweet_label = batch[4].squeeze(dim=1).float()
    input_text = input_text.to(device)
    history_label = history_label.to(device)
    discussion = discussion.to(device)
    technical_discussion = technical_discussion.to(device)
    tweet_label = tweet_label.to(device)
    return {
        "input_ids": input_text,
        "history_label": history_label,
        "discussion": discussion,
        "technical_discussion": technical_discussion,
        "tweet_label": tweet_label,
    }


def eval_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mode: str,
    epoch: int,
):
    if mode == "valid":
        name = "val result"
    else:
        name = "test result"
    with torch.no_grad():
        eval_predict = []
        eval_label = []
        eval_epoch_loss = 0
        for batch in dataloader:
            tweet_label = batch[4].squeeze().float()
            batch = move_to_cuda(batch=batch, model=model, device=device)
            predict, loss = model(**batch)
            eval_epoch_loss += loss.detach().to("cpu").item()
            eval_predict.append(predict.detach().to("cpu").numpy())
            eval_label.append(tweet_label.detach().to("cpu").numpy())
    return report_performance(
        np.concatenate(eval_label, axis=0),
        np.concatenate(eval_predict, axis=0),
        eval_epoch_loss,
        name,
        epoch_id=epoch,
        config=model_config,
    )


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

    if debug_test:
        model_config = {"alias": "debug_test_{}".format(T_dash), "bert_type": bert_type}
        num_epochs = 2

    train_dataloader = get_dataloader(
        model_config=model_config,
        T_dash=T_dash,
        split="train",
        length_technical=length_technical,
        batch_size=batch_size,
        prune=prune,
        debug_test=debug_test,
    )
    valid_dataloader = get_dataloader(
        model_config=model_config,
        T_dash=T_dash,
        split="valid",
        length_technical=length_technical,
        batch_size=batch_size,
        prune=prune,
        debug_test=debug_test,
    )
    test_dataloader = get_dataloader(
        model_config=model_config,
        T_dash=T_dash,
        split="test",
        length_technical=length_technical,
        batch_size=batch_size,
        prune=prune,
        debug_test=debug_test,
    )
    optimizer = AdamW(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        epoch_train_loss = 0
        model.train()
        for _, batch in tqdm(enumerate(train_dataloader)):
            batch = move_to_cuda(batch=batch, model=model, device=device)
            _, loss = model(**batch)
            epoch_train_loss += loss.detach().to("cpu").item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        eval_model(
            model=model,
            dataloader=valid_dataloader,
            device=device,
            mode="valid",
            epoch=epoch,
        )
        eval_model(
            model=model,
            dataloader=test_dataloader,
            device=device,
            mode="test",
            epoch=epoch,
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
