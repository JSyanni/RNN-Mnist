# coding=utf-8

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 判定GPU是否存在
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义超参数
# RNN的输入是一个序列，sequence_length为序列长度，input_size为序列每个长度。
sequence_length = 28
input_size = 28
# 定义RNN隐含单元的大小。
hidden_size = 128
# 定义rnn的层数
num_layers = 1
# 识别的类别数量
num_classes = 10
# 批的大小
batch_size = 64
# 定义迭代次数
num_epochs = 1
# 定义学习率
learning_rate = 0.001

# MNIST 数据集  
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False,
                                          transform=transforms.ToTensor())

# 构建数据管道， 使用自己的数据集请参考：https://blog.csdn.net/u014365862/article/details/80506147 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# 定义RNN（LSTM）
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# 定义模型
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# 损失函数和优化算法
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss = []
test_accs = []

# 训练模型 
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        # 前向传播+计算loss  
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 后向传播+调整参数 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        # 每100个batch打印一次数据    
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# 模型测试部分    
# 测试阶段不需要计算梯度，注意  
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# 保存模型参数   
torch.save(model.state_dict(), 'model.ckpt')


def draw_train_process2(title, iters, acc, label_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("loss", fontsize=20)
    plt.plot(iters, acc, color='red', label=label_acc)
    # plt.plot(iters, accs, color='green', label=lable_acc)
    plt.legend()
    plt.grid()

    plt.show()


def draw_train_process1(title, iters, costs, label_cost):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("loss", fontsize=20)
    plt.plot(iters, costs, color='green', label=label_cost)
    # plt.plot(iters, accs, color='green', label=lable_acc)
    plt.legend()
    plt.grid()

    plt.show()


def draw_train_process2(title, iters, acc, label_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("loss", fontsize=20)
    plt.plot(iters, acc, color='red', label=label_acc)
    # plt.plot(iters, accs, color='green', label=lable_acc)
    plt.legend()
    plt.grid()

    plt.show()


train_iters = range(len(train_loss))
draw_train_process1('training', train_iters, train_loss, 'training loss')
train_iters = range(len(test_accs))
draw_train_process2('training', train_iters, test_accs, 'testing acc')