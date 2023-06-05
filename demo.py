import sys
from itertools import groupby

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data_utils

from dataset import CapchaDataset
from model import CRNN

gpu = torch.device("cuda")

gru_hidden_size = 256
gru_num_layers = 2
cnn_output_height = 4


def demo(model, demo_ds):
    model.eval()
    demo_loader = torch.utils.data.DataLoader(demo_ds, batch_size=1)

    (x, y) = next(iter(demo_loader))
    y_pred = model(
        x.view(x.shape[0], 1, x.shape[1], x.shape[2]).cuda()
    )
    _, max_index = torch.max(y_pred, dim=2)

    raw_prediction = list(max_index.detach().cpu().numpy())
    prediction = torch.IntTensor(
        [c for c, _ in groupby(raw_prediction) if c != demo_ds.blank_label]
    )

    mpl.rcParams["font.size"] = 8
    plt.imshow(x[0], cmap="gray")
    mpl.rcParams["font.size"] = 18
    plt.gcf().text(x=0.1, y=0.1, s="Actual: " + str(y[0].numpy()))
    plt.gcf().text(x=0.1, y=0.2, s="Predicted: " + np.array_str(prediction.numpy().reshape(len(prediction))))
    plt.savefig(f"./output/demo.png")
    plt.show()


if __name__ == "__main__":
    demo_ds = CapchaDataset((4, 5), samples=1)

    model = CRNN(
        cnn_output_height, gru_hidden_size, gru_num_layers, demo_ds.num_classes
    ).to(gpu)
    model.load_state_dict(torch.load("./checkpoints/checkpoint_20.pt"))

    demo(model, demo_ds)