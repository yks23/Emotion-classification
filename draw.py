import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.model import MLPModel
from src.dataset import load_dataset_with_embeddings
model = MLPModel(input_dim=50, hidden_dim=128, output_dim=2, pooling='mean')
# 假设 model 已定义好，并且有一段数据 loader
model.eval()

# 用来存储各层 activations
activations = {}

# 定义 hook 函数
def get_activation(name):
    def hook(module, input, output):
        # 把 output detach 下来并存为 CPU 张量
        activations[name] = output.detach().cpu()
    return hook

# 1. 注册 hook（以你关心的几个中间层为例）
#    这里举例监控 model.bert.encoder.layer[0].output
model.fc0.register_forward_hook(get_activation('layer0_out'))
model.fc1.register_forward_hook(get_activation('layer1_out'))
model.fc2.register_forward_hook(get_activation('layer2_out'))
# 你也可以监控任何 nn.Module，例如所有 transformer block、某个 linear、某个 conv 等
loader,_,_=load_dataset_with_embeddings(
    "Dataset/train.txt", "Dataset/wiki_word2vec_50.bin", batch_size=32)

# 2. 用一批样本跑一次 forward，触发 hook
with torch.no_grad():
    for batch_x, batch_y, lengths in loader:
        if isinstance(batch_x, torch.Tensor):
            batch_x = batch_x.to("cpu")
        if lengths is not None:
            lengths = lengths.to("cpu")
        model(batch_x, lengths)  # 触发 hook

# 3. 拿到 activation，计算分布性质
act = activations['layer0_out']    # e.g. [batch_size, seq_len, hidden_dim]
flat = act.view(-1)                # 展平成一维

mean_val = flat.mean().item()
std_val  = flat.std().item()
min_val  = flat.min().item()
max_val  = flat.max().item()
sparsity = (flat.abs() < 0.5).float().mean().item()  # 绝对值小于阈值的比例

print(f"mean={mean_val:.4f}, std={std_val:.4f}, "
      f"min={min_val:.4f}, max={max_val:.4f}, "
      f"sparsity(@0.5)={sparsity*100:.2f}%")

# 4. 可视化直方图
plt.hist(flat.numpy(), bins=100, log=True)
plt.title('Layer0 Activation Distribution')
plt.xlabel('Activation value')
plt.ylabel('Count (log scale)')
plt.savefig('layer0_activation_distribution.png')
plt.show()
plt.close()
