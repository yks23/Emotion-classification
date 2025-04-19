import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import load_dataset_with_embeddings,load_dataset_with_str
from src.model import LSTMModel, GRUModel, CNNModel, MLPModel,BiGRUModel,BiLSTMModel,BertModel
import os
import wandb
import argparse
from test import test


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="lstm", help="model name")
    args.add_argument("--hidden_dim", type=int, default=128, help="hidden dim")
    args.add_argument("--output_dir", type=str, default="output", help="output dir")
    args.add_argument("--batch_size", type=int, default=32, help="batch size")
    args.add_argument("--num_epochs", type=int, default=30, help="number of epochs")
    args.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    result = args.parse_args()
    return result


# 测试部分
if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_path = "Dataset/train.txt"
    w2v_bin_path = "Dataset/wiki_word2vec_50.bin"
    # train dataloader embedding
    
    loader, word2idx, vectors = load_dataset_with_embeddings(
        data_path, w2v_bin_path, batch_size=args.batch_size
    )
    # train dataloader str
    if args.model == "bert":
        loader, _, _ = load_dataset_with_str(
            data_path, batch_size=args.batch_size
        )
    # valid dataloader
    valid_data_path = "Dataset/validation.txt"
    valid_loader, _, _ = load_dataset_with_embeddings(
        valid_data_path, w2v_bin_path, batch_size=args.batch_size
    )
    if args.model == "bert":
        valid_loader, _, _ = load_dataset_with_str(
            valid_data_path, batch_size=args.batch_size
        )
    # test dataloader
    test_data_path = "Dataset/test.txt"
    test_loader, _, _ = load_dataset_with_embeddings(
        test_data_path, w2v_bin_path, batch_size=args.batch_size
    )
    if args.model == "bert":
        test_loader, _, _ = load_dataset_with_str(
            test_data_path, batch_size=args.batch_size
        )

    # 创建模型
    embedding_dim = 50
    hidden_dim = args.hidden_dim  # 隐藏层维度
    output_dim = 2  # 二分类

    if args.model == "lstm":
        model = LSTMModel(embedding_dim, hidden_dim, output_dim)
    elif args.model == "gru":
        model = GRUModel(embedding_dim, hidden_dim, output_dim)
    elif args.model == "cnn":
        model = CNNModel(embedding_dim, num_filters=100, output_dim=output_dim)
    elif args.model == "mlp":
        model = MLPModel(
            input_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim
        )
    elif args.model == "bilstm":
        model = BiLSTMModel(embedding_dim, hidden_dim, output_dim)
    elif args.model == "bigru":
        model = BiGRUModel(embedding_dim, hidden_dim, output_dim)
    elif args.model == "bert":
        model = BertModel(bert_model=None, output_dim=output_dim)
    else:
        raise ValueError("Invalid model name. Choose 'lstm', 'gru', 'cnn', or 'mlp', 'bilstm', 'bigru', 'bert'.")

    # 使用 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # 打印当前使用的设备
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.model == "bert":
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    # 如果使用了预训练的BERT模型，可能需要调整学习率和权重衰减参数
    # 训练轮数
    num_epochs = args.num_epochs
    global_step = 0
    wandb.init(
        project="text_classification_param_compare", name=f"{args.output_dir}", config=args
    )

    # 训练循环
    for epoch in range(num_epochs):
        for batch_x, batch_y, lengths in loader:
            batch_y= batch_y.to(device)
            if isinstance(batch_x, torch.Tensor):
                batch_x = batch_x.to(device)
            if lengths is not None:
                lengths = lengths.to("cpu")
            global_step += 1
            # 训练模型
            optimizer.zero_grad()
            output = model(batch_x, lengths)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
            wandb.log({"loss": loss.item()}, step=global_step)
            wandb.log({"epoch": epoch + 1}, step=global_step)
            if global_step % 100 == 0:
                for n,p in model.named_parameters():
                    wandb.log({n: p.norm()}, step=global_step)
        # 验证模型
        val_loss = []
        for batch_x, batch_y, lengths in valid_loader:
            batch_y= batch_y.to(device)
            
            if isinstance(batch_x, torch.Tensor):
                batch_x = batch_x.to(device)
            
            if lengths is not None:
                lengths = lengths.to("cpu")
            with torch.no_grad():
                output = model(batch_x, lengths)
                val_loss.append(criterion(output, batch_y).item())

        # 测试模型并计算性能指标
        precision, recall, f1, accuracy = test(
            model, test_loader, nn.CrossEntropyLoss(), device
        )
        precision_val, recall_val, f1_val, accuracy_val = test(
            model, valid_loader, nn.CrossEntropyLoss(), device
        )
        # Log validation and performance metrics to wandb
        wandb.log({"val_loss": sum(val_loss) / len(val_loss)}, step=global_step)

        wandb.log({"precision_test": precision}, step=global_step)
        wandb.log({"recall_test": recall}, step=global_step)
        wandb.log({"f1_test": f1}, step=global_step)
        wandb.log({"accuracy_test": accuracy}, step=global_step)

        wandb.log({"precision_val": precision_val}, step=global_step)
        wandb.log({"recall_val": recall_val}, step=global_step)
        wandb.log({"f1_val": f1_val}, step=global_step)
        wandb.log({"accuracy_val": accuracy_val}, step=global_step)

        # 保存模型检查点
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, f"{args.model}_model_{epoch}.pth"),
        )
