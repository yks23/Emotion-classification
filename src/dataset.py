import torch
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
import transformers

def load_dataset_with_embeddings(data_path, w2v_bin_path, batch_size=32, max_vocab_size=None):
    # 1. 加载词向量
    wv = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    embedding_dim = wv.vector_size

    # 限制词表大小（可选）
    words = wv.index_to_key[:max_vocab_size] if max_vocab_size else wv.index_to_key
    word2idx = {word: idx for idx, word in enumerate(words)}
    vectors = torch.FloatTensor([wv[word] for word in words])

    # 添加 <UNK> 和 <PAD>
    unk_vec = torch.mean(vectors, dim=0, keepdim=True)
    pad_vec = torch.zeros(1, embedding_dim)
    vectors = torch.cat([vectors, unk_vec, pad_vec], dim=0)
    word2idx['<UNK>'] = len(word2idx)
    word2idx['<PAD>'] = len(word2idx)

    # 2. 自定义 Dataset
    class TextDataset(Dataset):
        def __init__(self, filepath, word2idx):
            self.samples = []
            self.word2idx = word2idx
            self.unk = word2idx.get('<UNK>', 0)
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    label = int(parts[0])
                    tokens = parts[1:]
                    indices = [self.word2idx.get(w, self.unk) for w in tokens]
                    self.samples.append((indices, label))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]  # 返回的是 (word idx list, label)

    # 3. Collate：直接转为词向量（batch, seq_len, embed_dim）
    def collate_fn(batch):
        idx_seqs, labels = zip(*batch)
        lengths = [len(seq) for seq in idx_seqs]
        pad_idx = word2idx['<PAD>']

        padded = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq, dtype=torch.long) for seq in idx_seqs],
            batch_first=True,
            padding_value=pad_idx
        )
        embedded = vectors[padded]  # (batch, seq_len, embed_dim)

        return embedded, torch.tensor(labels), torch.tensor(lengths)

    dataset = TextDataset(data_path, word2idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return dataloader, word2idx, vectors

def load_dataset_with_str(data_path, batch_size=32):
    # 1. 自定义 Dataset
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-chinese")
    class TextDataset(Dataset):
        def __init__(self, filepath):
            self.samples = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    label = int(parts[0])
                    tokens = "".join(parts[1:])
                    self.samples.append((tokens, label))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]
    def collate_fn(batch):
        strs, labels = zip(*batch)
        tokens = tokenizer(list(strs), padding=True, truncation=True, return_tensors='pt')
        return tokens, torch.tensor(labels), None
    # 2. 创建 DataLoader
    dataloader= DataLoader(TextDataset(data_path), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    return dataloader, None, None  # 返回None表示没有word2idx和vectors