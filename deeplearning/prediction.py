import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda:0" if torch.cuda.is_available() else "cpu"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


class Model3(nn.Module):

    def __init__(self, hidden_dim=128):
        super(Model3, self).__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.fn = nn.Linear(hidden_dim*2, 4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batchsize, max_len, emb_dim = x.shape
        output, (h, c) = self.lstm(x)
        output = self.relu(output[:,-5:,:])
        output = self.fn(output)

        return output

class LSTM_Module(nn.Module):

    def __init__(self):
        super(LSTM_Module, self).__init__()
        pretrain = torch.load("./parameters_twt.pt")
        self.scalar = pretrain["scalar"]
        self.model = pretrain["model_params"]

    def predict(self, inputs):
        # inputs:List<60, 12>
        inputs = np.array(inputs, dtype=np.float32)
        inputs[:, 2] = self.scalar.transform(inputs[:, 2].reshape(-1, 1)).flatten()
        inputs[:, 3] = self.scalar.transform(inputs[:, 3].reshape(-1, 1)).flatten()
        inputs[:, 4] = self.scalar.transform(inputs[:, 4].reshape(-1, 1)).flatten()
        inputs[:, 5] = self.scalar.transform(inputs[:, 5].reshape(-1, 1)).flatten()
        x = (torch.tensor(inputs, dtype=torch.float32).unsqueeze(0))[:, :, :6]    # (1, 60, 6)
        
        self.model.eval()
        outputs = self.model(x)  # (1, 5, 4)
        output = self.scalar.inverse_transform(outputs[:, :, -1].detach().numpy())  # next five minutes, only output close (1, 5)
        return output.flatten().tolist()


if __name__ == "__main__":
    mod = LSTM_Module()
    inputs = [[random.randint(40000, 41000) for i in range(12)] for j in range(60)]
    output = mod.predict(inputs)
    print(output)
