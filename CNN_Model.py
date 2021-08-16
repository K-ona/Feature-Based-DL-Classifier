import torch
from torch import nn
import torch.nn.functional as F
import math


class CNNNeuralNetwork(nn.Module):
    def __init__(self, model_num=5, input_size=17, in_channel=1, out_channel=20, block_out_channel=10,
                 kernel_size=3, stride=1, num_layers=2, dropout_p=0.1):

        super(CNNNeuralNetwork, self).__init__()
        # Conv2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.dropout = nn.Dropout(dropout_p)

        self.block_out_channel = block_out_channel
        # self.norm = LayerNorm(size)
        self.cnns = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                          kernel_size=kernel_size, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_channel, out_channels=block_out_channel,
                          kernel_size=kernel_size, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.Linear()
            )
            for i in range(model_num)])

        self.output_layer = nn.Sequential(
            nn.Linear(self.submodel_output_size * self.model_num, self.hidden_size),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Softmax(dim=1)
        )

    def attention(self, query, key, value, dropout=None):
        # Scaled Dot Product Attention
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attention = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attention = dropout(p_attention)
        return torch.matmul(p_attention, value), p_attention

    def forward(self, x):
        self.Atten_Weight = nn.Parameter(torch.rand((x.shape[0],
                                         self.block_out_channel * x.shape[-2] * x.shape[-1], self.model_num)),
                                         requires_grad=True).to(self.device)

        for i, cnn in enumerate(self.cnns):
            self.Atten_Weight[:, :, i] = cnn(x).view(x.shape[0], -1, self.model_num)
            # print(l[0].weight.grad_fn)

        attention_output, tmp = self.attention(self.Atten_Weight.transpose(-2, -1),
                                               self.Atten_Weight.transpose(-2, -1),
                                               self.Atten_Weight.transpose(-2, -1))
        return self.output_layer(attention_output.view(x.shape[0], -1))

# model = NeuralNetwork().to(device)
# print(model)
# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
#
# model(torch.rand(16, 20).to(device))
