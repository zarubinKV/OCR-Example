import sys
from itertools import groupby

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from colorama import Fore
from tqdm import tqdm

from dataset import CapchaDataset
from model import CRNN

gpu = torch.device("cuda")
epochs = 20

gru_hidden_size = 256
gru_num_layers = 2
cnn_output_height = 4
cnn_output_width = 9
digits_per_sequence = 5

model_save_path = "./checkpoints"


def train_one_epoch(model, criterion, optimizer, data_loader) -> None:
    model.train()
    train_correct = 0
    train_total = 0
    for x_train, y_train in tqdm(
            data_loader,
            position=0,
            leave=True,
            file=sys.stdout,
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET),
    ):
        batch_size = x_train.shape[0]  # x_train.shape == torch.Size([64, 28, 140])
        x_train = x_train.view(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
        optimizer.zero_grad()
        y_pred = model(x_train.cuda())
        input_lengths = torch.IntTensor(batch_size).fill_(cnn_output_width)
        target_lengths = torch.IntTensor([len(t) for t in y_train])
        loss = criterion(y_pred, y_train, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        _, max_index = torch.max(
            y_pred, dim=2
        )
        for i in range(batch_size):
            raw_prediction = list(
                max_index[:, i].detach().cpu().numpy()
            )
            prediction = torch.IntTensor(
                [c for c, _ in groupby(raw_prediction) if c != train_ds.blank_label]
            )
            if is_correct(prediction, y_train[i], train_ds.blank_label):
                train_correct += 1
            train_total += 1
    print(
        "TRAINING. Correct: ",
        train_correct,
        "/",
        train_total,
        "=",
        train_correct / train_total,
    )


def is_correct(prediction, y_true, blank):
    prediction = prediction.to(torch.int32)
    prediction = prediction[prediction != blank]
    y_true = y_true.to(torch.int32)
    y_true = y_true[y_true != blank]
    return len(prediction) == len(y_true) and torch.all(prediction.eq(y_true))


def evaluate(model, val_loader) -> float:
    model.eval()
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        for x_val, y_val in tqdm(
                val_loader,
                position=0,
                leave=True,
                file=sys.stdout,
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET),
        ):
            batch_size = x_val.shape[0]
            x_val = x_val.view(x_val.shape[0], 1, x_val.shape[1], x_val.shape[2])
            y_pred = model(x_val.cuda())
            input_lengths = torch.IntTensor(batch_size).fill_(cnn_output_width)
            target_lengths = torch.IntTensor([len(t) for t in y_val])
            criterion(y_pred, y_val, input_lengths, target_lengths)
            _, max_index = torch.max(y_pred, dim=2)
            for i in range(batch_size):
                raw_prediction = list(max_index[:, i].detach().cpu().numpy())
                prediction = torch.IntTensor(
                    [c for c, _ in groupby(raw_prediction) if c != train_ds.blank_label]
                )
                if is_correct(prediction, y_val[i], train_ds.blank_label):
                    val_correct += 1
                val_total += 1
        acc = val_correct / val_total
        print("TESTING. Correct: ", val_correct, "/", val_total, "=", acc)
    return acc


def test_model(model, test_ds, number_of_test_imgs: int = 10):
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=number_of_test_imgs)
    test_preds = []
    (x_test, y_test) = next(iter(test_loader))
    y_pred = model(
        x_test.view(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]).cuda()
    )
    _, max_index = torch.max(y_pred, dim=2)
    for i in range(x_test.shape[0]):
        raw_prediction = list(max_index[:, i].detach().cpu().numpy())
        prediction = torch.IntTensor(
            [c for c, _ in groupby(raw_prediction) if c != train_ds.blank_label]
        )
        test_preds.append(prediction)

    for j in range(len(x_test)):
        mpl.rcParams["font.size"] = 8
        plt.imshow(x_test[j], cmap="gray")
        mpl.rcParams["font.size"] = 18
        plt.gcf().text(x=0.1, y=0.1, s="Actual: " + str(y_test[j].numpy()))
        plt.gcf().text(x=0.1, y=0.2, s="Predicted: " + str(test_preds[j].numpy()))
        plt.savefig(f"./output/plot_{j}.png")
        # plt.show()


if __name__ == "__main__":
    train_ds = CapchaDataset((4, 5))
    test_ds = CapchaDataset((4, 5), samples=100)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64)
    val_loader = torch.utils.data.DataLoader(test_ds, batch_size=1)

    model = CRNN(
        cnn_output_height, gru_hidden_size, gru_num_layers, train_ds.num_classes
    ).to(gpu)
    # model.load_state_dict(torch.load("./checkpoints/checkpoint_20.pt"))

    criterion = nn.CTCLoss(
        blank=train_ds.blank_label, reduction="mean", zero_infinity=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    current_acc = 0
    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}/{epochs}")
        train_one_epoch(model, criterion, optimizer, train_loader)
        acc = evaluate(model, val_loader)
        if acc > current_acc:
            model_out_name = model_save_path + f"/checkpoint_{epoch}.pt"
            torch.save(model.state_dict(), model_out_name)

    test_model(model, test_ds)
