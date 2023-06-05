import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(self, cnn_output_height, gru_hidden_size, gru_num_layers, num_classes):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d((1, 2), stride=2)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(512)
        self.maxpool4 = nn.MaxPool2d((1, 2), stride=2)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=(2, 2), stride=1, padding=0)

        self.map_to_seq = nn.Linear(512 * 1, 64)

        self.rnn1 = nn.LSTM(64, gru_hidden_size , bidirectional=True)
        self.rnn2 = nn.LSTM(2 * gru_hidden_size , gru_hidden_size , bidirectional=True)

        self.fc = nn.Linear(gru_hidden_size * 2, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.leaky_relu(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = F.leaky_relu(out)
        out = self.maxpool2(out)
        out = self.conv3(out)
        out = F.leaky_relu(out)
        out = self.conv4(out)
        out = F.leaky_relu(out)
        out = self.maxpool3(out)
        out = self.conv5(out)
        out = self.norm1(out)
        out = F.leaky_relu(out)
        out = self.conv6(out)
        out = self.norm2(out)
        out = F.leaky_relu(out)
        out = self.maxpool3(out)
        out = self.conv7(out)
        out = F.leaky_relu(out)

        batch, channel, height, width = out.size()

        out = out.view(batch, channel * height, width)
        out = out.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(out)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)


        output = torch.stack(
            [F.log_softmax(self.fc(recurrent[i]), dim=-1) for i in range(recurrent.shape[0])]
        )
        return output