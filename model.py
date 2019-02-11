import torch
import torchvision.models as models
from torch import nn
from torch.autograd import Variable
from torch import tanh, sigmoid


class CombinedModel(nn.Module):

    def __init__(self, hidden_size, kernel_sizes, time_steps):
        super(CombinedModel, self).__init__()
        self.model_res = models.resnet50(pretrained=True)
        self.model_res.eval()
        self.model_res.fc = Identity()
        self.model_res.avgpool = Identity()
        self.model_CLSTMcell = ConvLSTMCell(2048, hidden_size, kernel_sizes)
        self.T = time_steps
        self.model_up_sample_cov = UpSampleConv(hidden_size)

    def forward(self, x):
        height, width = x.shape[2:4]
        x = self.model_res(x)
        x = x.view(x.shape[0], 2048, height//32, width//32)
        state = None
        for t in range(self.T):
            state = self.model_CLSTMcell(x, state)
        x, _ = state
        x = self.model_up_sample_cov(x)
        return x


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_sizes):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_sizes, padding=kernel_sizes // 2)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = sigmoid(in_gate)
        remember_gate = sigmoid(remember_gate)
        out_gate = sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * tanh(cell)

        return hidden, cell


class UpSampleConv(nn.Module):
    def __init__(self, channel_size, kernel_sizes=1):
        super().__init__()
        self.conv2 = nn.Conv2d(in_channels=channel_size, out_channels=1, kernel_size=kernel_sizes)
        self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)

    def forward(self, x):
        out = self.conv2(x)
        out = sigmoid(self.up_sample(out))
        return out


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.avgpool = nn.AvgPool2d((15, 20), stride=1)

    def forward(self, x):
        return x


def NSS(outputs, labels):
    means = torch.mean(outputs, 1, keepdim=True)
    stds = torch.std(outputs, 1, keepdim=True)
    eps = 1e-8
    Ns = torch.sum(labels, 1, keepdim=True)
    res = torch.sum(torch.sum((outputs - means)/(stds + eps)*labels, 1, keepdim=True)/Ns)/outputs.shape[0]
    return res


class CombinedLoss(torch.nn.Module):
    def __init__(self, w_MSE=0.33, w_KLDiv=0.33, w_NSS=0.33):
        super(CombinedLoss, self).__init__()
        self.w_MSE = w_MSE
        self.w_KLDiv = w_KLDiv
        self.w_NSS = w_NSS
        if w_MSE is not None:
            self.MSELoss = nn.MSELoss()
        if w_KLDiv is not None:
            self.KLDivLoss = nn.KLDivLoss()

    def forward(self, outputs, labels):
        outputs = outputs.view(outputs.shape[0], -1)
        labels = labels.view(labels.shape[0], -1)
        loss = None
        if self.w_MSE is not None:
            if loss is None:
                loss = self.w_MSE * self.MSELoss(outputs, labels)
            else:
                loss += self.w_MSE * self.MSELoss(outputs, labels)
        if self.w_KLDiv is not None:
            if loss is None:
                loss = self.w_KLDiv * self.KLDivLoss(outputs, labels)
            else:
                loss += self.w_KLDiv * self.KLDivLoss(outputs, labels)
        if self.w_NSS is not None:
            if loss is None:
                loss = -self.w_NSS * NSS(outputs, labels)
            else:
                loss += -self.w_NSS * NSS(outputs, labels)
        return loss
