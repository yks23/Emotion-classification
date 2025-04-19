import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import load_dataset_with_embeddings
from src.model import LSTMModel, GRUModel, CNNModel, MLPModel
import os
import wandb
import argparse



def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="lstm", help="model name")
    args.add_argument("--hidden_dim", type=int, default=128, help="hidden dim")
    args.add_argument("--output_dir", type=str, default="output", help="output dir")
    args.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    args.add_argument(
        "--test_data", type=str, required=True, help="Path to the test data"
    )
    result = args.parse_args()
    return result


def test(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    test_loss = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    with torch.no_grad():
        for batch_x, batch_y, lengths in test_loader:
            batch_y= batch_y.to(device)
            if isinstance(batch_x, torch.Tensor):
                batch_x = batch_x.to(device)
            if lengths is not None:
                lengths = lengths.to("cpu")
            output = model(batch_x, lengths)
            loss = criterion(output, batch_y)
            test_loss += loss.item()

            # 统计 TP, FP, TN, FN
            _, predicted = torch.max(output, 1)
            for i in range(batch_y.size(0)):  # 遍历当前batch中的每个样本
                if batch_y[i].item() == 1:
                    if predicted[i].item() == 1:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if predicted[i].item() == 0:
                        TN += 1
                    else:
                        FP += 1

            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    # 计算精度、召回率和F1分数
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    F1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) != 0
        else 0
    )
    accuracy = correct / total if total != 0 else 0
    model.train()
    return precision, recall, F1_score, accuracy


# 主函数
if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()
    w2v_bin_path = "Dataset/wiki_word2vec_50.bin"
    test_loader, _, _ = load_dataset_with_embeddings(
        args.test_data, w2v_bin_path, batch_size=1
    )

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
        model = MLPModel(input_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    else:
        raise ValueError("Invalid model name. Choose 'lstm', 'gru', 'cnn', or 'mlp'.")

    # 使用 CPU 或 GPU
    device = torch.device("cpu")
    model.to(device)
    checkpoint = torch.load(args.model_checkpoint)
    model.load_state_dict(checkpoint)

    # 测试时的损失函数
    criterion = nn.CrossEntropyLoss()

    # 在测试集上测试模型
    precision, recall, f1_score,accuracy = test(model, test_loader, criterion, device)

    # 创建输出目录
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存测试结果
    target_result_path = os.path.join(
        output_dir, f"result_{os.path.basename(args.model_checkpoint)}.txt"
    )
    with open(target_result_path, "w") as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1_score:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")

    # 打印结果
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
