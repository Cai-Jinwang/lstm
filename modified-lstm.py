# _*_ encoding:utf-8 _*_
__author__ = 'Cai Jinwang'
__date__ = '2024/9/9 0:40'
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import jieba
import time

# 设置随机种子以确保可复现性
torch.manual_seed(0)
np.random.seed(0)

# 读取数据
file_path = './mnt/data/online_shopping_10_cats.csv'
data = pd.read_csv(file_path).astype(str)

# 使用LabelEncoder对标签进行编码
label_encoder = LabelEncoder()
data.loc[:, 'label'] = label_encoder.fit_transform(data['cat'])  # 使用.loc来赋值
# data['label'] = label_encoder.fit_transform(data['label'])

#%% 文本预处理
def preprocess_text(text):
    # 分词
    tokens = jieba.lcut(text)
    # 返回分词后的结果
    return ' '.join(tokens)

data['review'] = data['review'].apply(preprocess_text)

# 构建词汇表
vocab = set(' '.join(data['review']).split())
word_to_ix = {word: i + 1 for i, word in enumerate(vocab)}
word_to_ix['<PAD>'] = 0  # 添加padding标记

# 文本到索引的转换，并进行填充
def text_to_sequence(text, max_len):
    tokens = text.split()
    sequence = [word_to_ix.get(word, 0) for word in tokens]
    # 填充或截断序列
    if len(sequence) < max_len:
        sequence += [word_to_ix['<PAD>']] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    return sequence

# 确定最大序列长度
max_sequence_length = max([len(text.split()) for text in data['review']])

# 划分训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=0)

# 对训练集和测试集进行深拷贝，避免 SettingWithCopyWarning
train_data = train_data.copy()
test_data = test_data.copy()

# 使用 .loc 进行安全的赋值操作
train_data.loc[:, 'review_seq'] = train_data['review'].apply(lambda x: text_to_sequence(x, max_sequence_length))
test_data.loc[:, 'review_seq'] = test_data['review'].apply(lambda x: text_to_sequence(x, max_sequence_length))


# 使用.loc进行赋值
train_data.loc[:, 'review_seq'] = train_data['review'].apply(lambda x: text_to_sequence(x, max_sequence_length))
test_data.loc[:, 'review_seq'] = test_data['review'].apply(lambda x: text_to_sequence(x, max_sequence_length))

# train_data['review_seq'] = train_data['review'].apply(lambda x: text_to_sequence(x, max_sequence_length))
# test_data['review_seq'] = test_data['review'].apply(lambda x: text_to_sequence(x, max_sequence_length))

#%% 定义数据集类
class ReviewDataset(Dataset):
    def __init__(self, reviews, labels):
        self.reviews = reviews
        self.labels = labels

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]
        return torch.tensor(review, dtype=torch.long), torch.tensor(label, dtype=torch.long)

#%% 创建数据加载器
print('创建数据加载器')
train_dataset = ReviewDataset(train_data['review_seq'].tolist(), train_data['label'].tolist())
test_dataset = ReviewDataset(test_data['review_seq'].tolist(), test_data['label'].tolist())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#%%# 检查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#%% 定义LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out

#%% 初始化模型
vocab_size = len(word_to_ix) + 1
embedding_dim = 100
hidden_dim = 128
model = LSTMClassifier(embedding_dim, hidden_dim, vocab_size).to(device)  # 将模型移动到GPU

#%% 定义损失函数和优化器
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#%% 训练模型
num_epochs = 5
print_interval = 10  # 每隔10个batch打印一次

for epoch in range(num_epochs):
    start_time = time.time()  # 记录epoch开始时间
    running_loss = 0.0  # 累积损失

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        batch_start_time = time.time()  # 记录每个batch的开始时间

        # 将数据和标签移动到 GPU
        inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = loss_function(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 每隔 print_interval 批次打印一次信息
        if (batch_idx + 1) % print_interval == 0:
            batch_time = time.time() - batch_start_time
            print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)} - '
                  f'Loss: {loss.item():.4f} - Time: {batch_time:.4f} sec', flush=True)

    # 每个 epoch 处理完后的耗时
    epoch_time = time.time() - start_time
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss:.4f} - Epoch Time: {epoch_time:.4f} sec', flush=True)

#%% 模型评估函数
print('开始进行模型评估')
def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_time = 0  # 总时间
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            batch_start_time = time.time()  # 记录每个batch的开始时间

            # 将数据移动到 GPU
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)

            outputs = model(inputs)
            predicted = (outputs > 0.5).float()

            all_preds.extend(predicted.view(-1).tolist())
            all_labels.extend(labels.view(-1).tolist())

            # 记录每个batch的处理时间
            batch_time = time.time() - batch_start_time
            total_time += batch_time

            if (batch_idx + 1) % print_interval == 0:
                print(f'Test Batch {batch_idx + 1}/{len(data_loader)} - Time: {batch_time:.4f} sec', flush=True)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    print(f'Total evaluation time: {total_time:.4f} sec', flush=True)
    return accuracy, precision, recall, f1

# 测试模型
accuracy, precision, recall, f1 = evaluate_model(model, test_loader)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

"""
这段代码实现了一个基于 LSTM（长短期记忆网络）的文本分类模型，使用 PyTorch 框架。主要步骤如下：

1. **数据加载与预处理**：
   - 从 `online_shopping_10_cats.csv` 文件中读取数据集。
   - 使用 Jieba 对中文文本进行分词处理。
   - 使用 `LabelEncoder` 对标签进行编码，将类别转换为数字。
   - 根据数据集构建词汇表，将文本转化为对应的索引序列，并对序列进行填充或截断。

2. **数据集与 DataLoader**：
   - 定义了一个自定义的 `ReviewDataset` 数据集类，继承自 PyTorch 的 `Dataset`，用来加载评论文本及其对应的标签。
   - 使用 `DataLoader` 创建训练集和测试集的数据加载器，将数据分成批次，以便模型进行训练和测试。

3. **LSTM 模型**：
   - 定义了一个 LSTM 分类模型 `LSTMClassifier`，模型包括嵌入层（embedding layer）、LSTM 层和一个全连接层（fully connected layer）来进行分类预测。
   - 通过 LSTM 处理后的输出，使用全连接层生成最终的预测结果。

4. **训练模型**：
   - 使用 Adam 优化器和二元交叉熵损失函数 `BCEWithLogitsLoss` 进行模型的训练，适用于二分类任务。
   - 训练过程中每个 epoch 都会打印损失值。

5. **模型评估**：
   - 定义了 `evaluate_model` 函数，用于计算模型在测试集上的准确率（accuracy）、精确率（precision）、召回率（recall）和 F1 分数。
   - 最后输出模型的各项评估指标。

### 改进建议：
- 可以加入**早停法（Early Stopping）**，避免过拟合。
- 如果数据集类别不平衡，可以考虑使用**损失加权**或**过采样/欠采样**方法。
- 可以尝试调整**模型的超参数**（如嵌入维度、隐藏层维度）或使用更复杂的网络结构来提升模型性能。

如果你需要运行或修改这段代码的帮助，请告诉我！
"""