import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torchinfo import summary
from tqdm import tqdm

from models.PDRSI import PDRSI
from utility.dataset import get_dataloader #! add get_dataloader and remove MyDatasetWithBothDiscussion and get_data
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
    #! 個人的ではあるが，GPUあるのにCPUでやりたい人はCUDA_VISIBLE_DEVICE=0 python model_run.py
    #! で実行するのでこれはなくても良いと思う
    use_gpu = True

    if debug_test:
        model_config = {"alias": "debug_test_{}".format(T_dash), "bert_type": bert_type}
        num_epochs = 2

    #! train, valid, testごとに毎行書くのは冗長
    #! datasetはdataloaderに入力する以外使わないのでget_dataloaderに押し込めた
    #! テストとかきちんとするなら分けるべきではあると思う
    train_dataloader = get_dataloader(
        model_config=model_config,
        T_dash=T_dash,
        split="train",
        length_technical=length_technical,
        batch_size=batch_size,
        prune=prune,
        debug_test=debug_test
    )
    valid_dataloader = get_dataloader(
        model_config=model_config,
        T_dash=T_dash,
        split="valid",
        length_technical=length_technical,
        batch_size=batch_size,
        prune=prune,
        debug_test=debug_test
    )
    test_dataloader = get_dataloader(
        model_config=model_config,
        T_dash=T_dash,
        split="test",
        length_technical=length_technical,
        batch_size=batch_size,
        prune=prune,
        debug_test=debug_test
    )
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
            #! squeezeやfloatの処理をdatasetや前処理に入れる+lossの処理までモデルの実装に入れると
            #! 以下のように書ける
            """
            def move_to_cuda(
                batch: Dict[str, torch.Tensor], device: torch.device
            ) -> Dict[str, torch.Tensor]:
                for k, v in batch.items():
                    batch[k] = v.to(device)
                return batch
            
            batch = move_to_cuda(batch, device)
            predict, loss = model(**batch)
            """
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
        #! validationもbackwardせず勾配が必要ないのでwith torch.no_grad()の中に入れて良い
        #! またvalidationとtestはほぼ同じコードなはずなので，関数化することで冗長さが減らせると思う(これは人による)
        """
        def eval_model(model: nn.Module, dataloader: DataLoader, device: torch.device):
            with torch.no_grad():
                for batch in loader:
                    # モデルに入力したり評価
            return report_performance(色々入力)
        """
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
            #! 以下3行，to("cpu")しなくてもcpuに渡る気がする
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
                #! remove criterion
                # loss = criterion(predict, tweet_label)
                #! 以下3行，to("cpu")しなくてもcpuに渡る気がする
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
