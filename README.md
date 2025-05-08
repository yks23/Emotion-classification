# Get Started
依赖在requirements.txt
新建环境然后：
```bash
pip install -r requirements.txt
```
# train
train.sh为训练脚本
你需要修改其中的环境变量以按照你的想法进行训练，具体参看脚本注释
然后:
```bash
bash train.sh
```
# test
test.sh为测试脚本
你需要修改其中的环境变量以按照你的想法进行指标测试，具体参看脚本注释
然后:
```bash
bash test.sh
```
# checkpoint
拿了一些checkpoint上传，不代表最好效果
参看./checkpoint

# 源码指引
- train.py为训练代码
- test.py为测试代码
- src/model.py为模型定义
- src/dataset.py为数据集
# 可视化
支持wandb