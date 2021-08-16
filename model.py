import torch
from torch import nn
import torch.nn.functional as F
import math


# data_path = './data/Cleaned_data.csv'
#
# # Get cpu or gpu device for training.
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using {} device".format(device))


# device = 'cpu'
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, device='cuda', model_num=5, input_size=17, hidden_size=64,
                 output_size=5, submodel_output_size=16, num_layers=2, dropout_p=0.1,
                 batch_size=32, atten_flag=True):
        #
        #
        super(NeuralNetwork, self).__init__()
        self.model_num = model_num
        self.input_size = input_size
        self.output_size = output_size
        self.submodel_output_size = submodel_output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = device
        self.atten_flag = atten_flag

        self.dropout = nn.Dropout(dropout_p)
        # self.norm = LayerNorm(size)
        self.linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.Dropout(dropout_p),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.submodel_output_size))
            for i in range(self.model_num)])

        self.output_layer = nn.Sequential(
            nn.Linear(self.submodel_output_size * self.model_num, self.hidden_size),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
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
        self.Atten_Weight = nn.Parameter(torch.rand((x.shape[0], self.submodel_output_size, self.model_num)),
                                         requires_grad=True).to(self.device)

        for i, l in enumerate(self.linears):
            self.Atten_Weight[:, :, i] = l(x)
            # print(l[0].weight.grad_fn)
        if self.atten_flag:
            attention_output, tmp = self.attention(self.Atten_Weight.transpose(-2, -1),
                                                   self.Atten_Weight.transpose(-2, -1),
                                                   self.Atten_Weight.transpose(-2, -1))
        else:
            attention_output = self.Atten_Weight
        return self.output_layer(attention_output.view(x.shape[0], -1))

# model = NeuralNetwork().to(device)
# print(model)
# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
#
# model(torch.rand(16, 20).to(device))
