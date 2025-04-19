import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        # 使用 pack_padded_sequence 来忽略填充部分
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed_input)
        # 输出的 hn 是 (num_layers, batch_size, hidden_dim)，取最后一层的输出
        output = self.fc(hn[-1])
        return output
    
class GRUModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        # 使用 pack_padded_sequence 来忽略填充部分
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hn = self.gru(packed_input)
        # 输出的 hn 是 (num_layers, batch_size, hidden_dim)，取最后一层的输出
        output = self.fc(hn[-1])
        return output
    
class CNNModel(nn.Module):
    def __init__(self, embedding_dim, num_filters, output_dim, kernel_sizes=(2, 3, 4, 5)):
        super(CNNModel, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(k, embedding_dim))
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_dim)

    def forward(self, x, lengths=None):
        # x: (batch, seq_len, embedding_dim)
        x = x.unsqueeze(1)  # (batch, 1, seq_len, embedding_dim)
        conv_outs = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch, num_filters, seq_len - k + 1), ...]
        pooled = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outs]  # [(batch, num_filters), ...]
        cat = torch.cat(pooled, dim=1)  # (batch, num_filters * len(kernel_sizes))
        cat = self.dropout(cat)
        return self.fc(cat)
    
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pooling='mean'):
        super(MLPModel, self).__init__()
        self.pooling = pooling
        self.fc0 = nn.Linear(input_dim, input_dim)  # 线性变换
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths=None):
        # x: [batch_size, seq_len, embedding_dim]
        x= self.fc0(x)
        x = F.relu(x)  # 激活函数
        
        if self.pooling == 'max':
            x, _ = torch.max(x, dim=1)  # [batch_size, embedding_dim]
        elif self.pooling == 'mean':
            x = torch.mean(x, dim=1)    # [batch_size, embedding_dim]
        else:
            raise ValueError("Unsupported pooling type. Use 'max' or 'mean'.")

        x = F.relu(self.fc1(x))         # [batch_size, hidden_dim]
        x = self.dropout(x)
        x = self.fc2(x)                 # [batch_size, output_dim]

        return x

class BiLSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # 双向LSTM的输出维度是hidden_dim * 2

    def forward(self, x, lengths):
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed_input)
        h_forward = hn[0]       # (batch, hidden_dim)
        h_backward = hn[1]      # (batch, hidden_dim)
        # 拼接起来就是 (batch, hidden_dim * 2)
        h_concat = torch.cat([h_forward, h_backward], dim=1)
        # 再过一层全连线做分类
        output = self.fc(h_concat)   # (batch, output_dim)
        return output


class BiGRUModel(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(BiGRUModel, self).__init__()
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # 双向GRU的输出维度是hidden_dim * 2

    def forward(self, x, lengths):
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hn = self.gru(packed_input)
        h_forward = hn[0]       # (batch, hidden_dim)
        h_backward = hn[1]      # (batch, hidden_dim)
        # 拼接起来就是 (batch, hidden_dim * 2)
        h_concat = torch.cat([h_forward, h_backward], dim=1)
        # 再过一层全连线做分类
        output = self.fc(h_concat)   # (batch, output_dim)
        return output
    
class BertModel(nn.Module):
    def __init__(self,bert_model=None,output_dim=2):
        super(BertModel, self).__init__()
        if bert_model is None:
            bert_model = transformers.BertModel.from_pretrained("bert-base-chinese")
        self.bert = bert_model
        self.fc = nn.Linear(768, output_dim)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x, lengths=None):
        # x: [batch_size, seq_len]
        for k, v in x.items():
            x[k] = v.to(self.bert.device)
        # 将tokens移动到GPU（如果可用）
        print(x['input_ids'][0],x['attention_mask'][0],x['token_type_ids'][0])
        outputs = self.bert(**x)
        # 使用pooler_output作为句子表示
        x = outputs['last_hidden_state'][:, 0, :]  # [batch_size, 768]
        x = self.dropout(x)
        x = self.fc(x)
        return x