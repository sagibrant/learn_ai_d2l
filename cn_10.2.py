import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt


# 10.2.1. 生成数据集
n_train = 50  # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5)   # 排序后的训练样本

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
x_test = torch.arange(0, 5, 0.1)  # 测试样本
y_truth = f(x_test)  # 测试样本的真实输出
n_test = len(x_test)  # 测试样本数
print("n_test: ", n_test)

# 下面的函数将绘制所有的训练样本
# def plot_kernel_reg(y_hat):
#     d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
#              xlim=[0, 5], ylim=[-1, 5])
#     d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);

def plot_kernel_reg(y_hat):
    plt.figure()
    plt.plot(x_test.numpy(), y_truth.numpy(), label='Truth')
    plt.plot(x_test.numpy(), y_hat.detach().numpy(), label='Pred')
    plt.scatter(x_train.numpy(), y_train.numpy(), alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0, 5])
    plt.ylim([-1, 5])
    plt.legend()
    plt.show()

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    plt.show()


# 10.2.2. 平均汇聚
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)

# 10.2.3. 非参数注意力汇聚
# X_repeat的形状:(n_test,n_train),
# 每一行都包含着相同的测试输入（例如：同样的查询）
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
# x_train包含着键。attention_weights的形状：(n_test,n_train),
# 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
y_hat = torch.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)

show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')


# 10.2.4. 带参数注意力汇聚
# 10.2.4.1. 批量矩阵乘法
# (n,a,b) bmm (n,b,c) = (n,a,c)
X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
torch.bmm(X, Y).shape
# torch.Size([2, 1, 6])

# 在注意力机制的背景中，我们可以使用小批量矩阵乘法来计算小批量数据中的加权平均值。
weights = torch.ones((2, 10)) * 0.1
values = torch.arange(20.0).reshape((2, 10))
torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1)) # .unsqueeze(-1) => torch.Size([2, 10, 1])
# tensor([[[ 4.5000]],

#         [[14.5000]]])

# 10.2.4.2. 定义模型
# 基于 (10.2.7)中的 带参数的注意力汇聚，使用小批量矩阵乘法， 定义Nadaraya-Watson核回归的带参数版本为
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))
    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # values的形状为(查询个数，“键－值”对个数)
        # (50, 1, 49) @ (50, 49, 1) → (50, 1, 1)
        y_hat = torch.bmm(self.attention_weights.unsqueeze(1), # shape: (n, 1, m)
                         values.unsqueeze(-1)).reshape(-1)     # shape: (n, m, 1)
        print("NWKernelRegression", self.attention_weights.shape, values.shape, y_hat)
        return y_hat
        
    

# 10.2.4.3. 训练
# 这段代码的目的是为了在训练过程中构造每个样本对应的“其他所有样本”作为 key/value，从而进行留一（leave-one-out）注意力训练，避免模型用自己预测自己。
# .repeat((n_train, 1))为每个样本构造一个“键集合”，也就是所有训练样本都被当成 key——包括自己。
# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = x_train.repeat((n_train, 1))
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))
# keys的形状:('n_train'，'n_train'-1)
# 用布尔掩码的方式排除对角线元素。也就是排除样本本身，仅保留其它样本。
# X_tile[(1 - torch.eye(n_train)).type(torch.bool)].shape = torch.Size([2450])
# .reshape((n_train, -1)) 即恢复成二维结构：每一行是 x_train 除了第 i 个元素以外的所有元素。
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
# values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

# 定义 网络
net = NWKernelRegression()
# 定义 损失 用于计算 均方误差（Mean Squared Error, MSE） 的一种损失函数, 在 神经网络训练 中，MSE 用来衡量模型输出与真实标签之间的差距，作为 反向传播的依据。
loss = nn.MSELoss(reduction='none') 
# 创建优化器: PyTorch 中的一个 优化器（Optimizer），SGD 代表 随机梯度下降（Stochastic Gradient Descent）。
# 在神经网络训练过程中，优化器负责根据 损失函数的梯度 来更新模型参数，使损失逐渐减小，模型预测越来越准确。
trainer = torch.optim.SGD(net.parameters(), lr=0.5) 
epochs = []
losses = []
# 训练带参数的注意力汇聚模型时，使用平方损失函数和随机梯度下降。
for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    curr_loss = float(l.sum())
    print(f'epoch {epoch + 1}, loss {curr_loss:.6f}')
    epochs.append(epoch + 1)
    losses.append(curr_loss)

# 绘制训练过程中的 loss 曲线
def plot_loss(epochs, losses):
    plt.figure()
    plt.plot(epochs, losses, marker='o')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()

plot_loss(epochs, losses)

# 在尝试拟合带噪声的训练数据时， 预测结果绘制的线不如之前非参数模型的平滑。
# keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
keys = x_train.repeat((n_test, 1))
# value的形状:(n_test，n_train)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)

show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')