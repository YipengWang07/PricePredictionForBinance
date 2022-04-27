import sklearn
import sklearn.preprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model3(nn.Module):

    def __init__(self, hidden_dim=128):
        super(Model3, self).__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=hidden_dim, num_layers=2, bidirectional=True, batch_first=True,
                            dropout=0.2)
        self.fn = nn.Linear(hidden_dim * 2, 4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batchsize, max_len, emb_dim = x.shape
        output, (h, c) = self.lstm(x)
        output = self.relu(output[:, -5:, :])
        output = self.fn(output)

        return output


class LSTM_Module(nn.Module):

    def __init__(self):
        super(LSTM_Module, self).__init__()
        pretrain = torch.load("parameters_twt.pt")
        self.scalar = pretrain["scalar"]
        self.model = pretrain["model_params"]

    def predict(self, inputs):
        # inputs: 1-D List[60*12]
        inputs = [inputs[i] for i in range(len(inputs)) if i % 12 != 0]
        inputs = np.array(inputs, dtype=np.float32).reshape(60, 11)
        inputs[:, 2] = self.scalar.transform(inputs[:, 2].reshape(-1, 1)).flatten()
        inputs[:, 3] = self.scalar.transform(inputs[:, 3].reshape(-1, 1)).flatten()
        inputs[:, 4] = self.scalar.transform(inputs[:, 4].reshape(-1, 1)).flatten()
        inputs[:, 5] = self.scalar.transform(inputs[:, 5].reshape(-1, 1)).flatten()
        x = (torch.tensor(inputs, dtype=torch.float32).unsqueeze(0))[:, :, :6]  # (1, 60, 6)

        self.model.eval()
        outputs = self.model(x)  # (1, 5, 4)
        output = self.scalar.inverse_transform(
            outputs[:, :, -1].detach().numpy())  # next five minutes, only output close (1, 5)
        return output.flatten().tolist()


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attn_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, Q, K, V):
        attn = F.softmax(torch.matmul(Q, K.permute(0, 2, 1)) / (K.shape[1] ** 0.5), dim=-1)
        # attn = self.dropout(attn)
        output = torch.matmul(attn, V)
        return output, attn

class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, input_dim, hidden_size=128, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.input_dim = input_dim

        self.w_Q = nn.Linear(input_dim, hidden_size*2, bias=False)
        self.w_K = nn.Linear(input_dim, hidden_size*2, bias=False)
        self.w_V = nn.Linear(input_dim, hidden_size*2, bias=False)
        self.attention = ScaledDotProductAttention()
        self.ln = nn.LayerNorm(hidden_size*2, eps=1e-6)
        self.fn = nn.Linear(hidden_size*2, hidden_size*2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        # input: (B, length, d1)
        Q = self.w_Q(inputs)
        K = self.w_K(inputs)
        V = self.w_V(inputs)
        lst = []
        feat_dim = V.shape[2] // self.n_head

        for i in range(self.n_head):
            out, att = self.attention(Q[:, :, (i*feat_dim):((i+1)*feat_dim)],
                                      K[:, :, (i*feat_dim):((i+1)*feat_dim)],
                                      V[:, :, (i*feat_dim):((i+1)*feat_dim)])
            lst.append(out)
        output = torch.cat(lst, dim=-1)
        output = self.fn(output)
        output = self.ln(inputs + output)

        return output

class Model4(nn.Module):
    def __init__(self, hidden_dim=128):
        super(Model4, self).__init__()

        self.lstm = nn.LSTM(input_size=6, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True, dropout=0.2)
        self.transformer_encoder_layer1 = MultiHeadAttention(n_head=2, input_dim=hidden_dim*2)
        self.transformer_encoder_layer2 = MultiHeadAttention(n_head=2, input_dim=hidden_dim*2)

        self.fn = nn.Linear(hidden_dim*2, 4)
        self.relu = nn.ReLU()


    def forward(self, x):
        x, (hn, cn) = self.lstm(x)  # (B, length, hidden_size*2)

        x = self.transformer_encoder_layer1(x)
        # x = self.transformer_encoder_layer2(x)
        output = self.relu(x[:,-5:,:])
        output = self.fn(output)
        return output


class Transformer_Module(nn.Module):
    def __init__(self):
        super(Transformer_Module, self).__init__()
        pretrain = torch.load("transformer_twt.pt")
        self.scalar = pretrain["scalar"]
        self.model = pretrain["model_params"]

    def predict(self, inputs):
        # inputs: 1-D List[60*12]
        inputs = [inputs[i] for i in range(len(inputs)) if i % 12 != 0]
        inputs = np.array(inputs, dtype=np.float32).reshape(60, 11)
        inputs[:, 2] = self.scalar.transform(inputs[:, 2].reshape(-1, 1)).flatten()
        inputs[:, 3] = self.scalar.transform(inputs[:, 3].reshape(-1, 1)).flatten()
        inputs[:, 4] = self.scalar.transform(inputs[:, 4].reshape(-1, 1)).flatten()
        inputs[:, 5] = self.scalar.transform(inputs[:, 5].reshape(-1, 1)).flatten()
        x = (torch.tensor(inputs, dtype=torch.float32).unsqueeze(0))[:, :, :6]  # (1, 60, 6)

        self.model.eval()
        outputs = self.model(x)  # (1, 5, 4)
        output = self.scalar.inverse_transform(
            outputs[:, :, -1].detach().numpy())  # next five minutes, only output close (1, 5)
        return output.flatten().tolist()


if __name__ == "__main__":
    import random
    mod = LSTM_Module()
    inputs = [random.randint(36000, 41000) for j in range(720)]
    # lstm
    output = mod.predict(inputs)
    print(output)
    # transformer
    mod2 = Transformer_Module()
    output2 = mod2.predict(inputs)
    print(output2)
