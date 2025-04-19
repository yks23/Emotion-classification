import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=50, hidden_size=128, num_layers=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, 2)  # 输出为2类

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)        # hn: (num_layers, batch, hidden_size)
        final_hidden = hn[-1]               # 取最后一层的输出
        logits = self.fc(final_hidden)
        return logits


class GRUClassifier(nn.Module):
    def __init__(self, input_size=50, hidden_size=128, num_layers=2, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, hn = self.gru(x)               # hn: (num_layers, batch, hidden_size)
        final_hidden = hn[-1]
        logits = self.fc(final_hidden)
        return logits
