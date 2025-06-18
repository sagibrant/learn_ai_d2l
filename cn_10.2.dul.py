import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

n_train = 50  # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5)   # 排序后的训练样本
print("x_train", x_train)
# torch.return_types.sort(
# values=tensor([0.1146, 0.1507, 0.3058, 0.4619, 0.5368, 0.5726, 0.6321, 0.6594, 0.6670,
#         0.6795, 0.7099, 0.7121, 0.8399, 1.0778, 1.1267, 1.1280, 1.3374, 1.4459,
#         1.4869, 1.5863, 1.6840, 1.7621, 1.9275, 2.0538, 2.0994, 2.1629, 2.3369,
#         2.3895, 2.4855, 2.4859, 2.4892, 2.6768, 2.6844, 3.0824, 3.1863, 3.2478,
#         3.6201, 3.6273, 3.8537, 4.0954, 4.1021, 4.1834, 4.2285, 4.4218, 4.6559,
#         4.6781, 4.7703, 4.7759, 4.9064, 4.9873]),
# indices=tensor([ 3, 19, 26, 10,  7, 37, 17, 23, 31, 44,  4,  5, 13, 16, 30, 22, 14, 47,
#         33, 15,  1, 35,  0, 39, 27, 32, 21, 42,  6,  9, 28, 38, 36,  2, 49, 45,
#         12, 29, 46, 25, 41, 34, 24, 11, 43, 48, 18,  8, 20, 40]))

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
print("y_train", y_train)
# tensor([0.7101, 0.4153, 1.1836, 2.5265, 2.1099, 1.0333, 1.7749, 1.8142, 2.2930,
#         2.8175, 1.8767, 2.3717, 1.8286, 3.1406, 2.8845, 3.1388, 2.9404, 3.6754,
#         2.7916, 3.3434, 2.8504, 3.7335, 3.0239, 5.0891, 3.5595, 2.9674, 3.4481,
#         3.1520, 3.6359, 3.9694, 3.2625, 3.5241, 3.1020, 1.5505, 3.6879, 2.4964,
#         1.9464, 2.3617, 1.6860, 0.4968, 0.8882, 0.6689, 1.4591, 1.7458, 1.3454,
#         0.7427, 1.6947, 0.9780, 1.8477, 0.7847])

x_test = torch.arange(0, 5, 0.1)  # 测试样本
print("x_test", x_test)
# tensor([0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000,
#         0.9000, 1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000, 1.6000, 1.7000,
#         1.8000, 1.9000, 2.0000, 2.1000, 2.2000, 2.3000, 2.4000, 2.5000, 2.6000,
#         2.7000, 2.8000, 2.9000, 3.0000, 3.1000, 3.2000, 3.3000, 3.4000, 3.5000,
#         3.6000, 3.7000, 3.8000, 3.9000, 4.0000, 4.1000, 4.2000, 4.3000, 4.4000,
#         4.5000, 4.6000, 4.7000, 4.8000, 4.9000])
# torch.Size([50])

y_truth = f(x_test)  # 测试样本的真实输出
print("y_truth", y_truth)
# tensor([0.0000, 0.3582, 0.6733, 0.9727, 1.2593, 1.5332, 1.7938, 2.0402, 2.2712,
#         2.4858, 2.6829, 2.8616, 3.0211, 3.1607, 3.2798, 3.3782, 3.4556, 3.5122,
#         3.5481, 3.5637, 3.5597, 3.5368, 3.4960, 3.4385, 3.3654, 3.2783, 3.1787,
#         3.0683, 2.9489, 2.8223, 2.6905, 2.5554, 2.4191, 2.2835, 2.1508, 2.0227,
#         1.9013, 1.7885, 1.6858, 1.5951, 1.5178, 1.4554, 1.4089, 1.3797, 1.3684,
#         1.3759, 1.4027, 1.4490, 1.5151, 1.6009])
# torch.Size([50])

n_test = len(x_test)  # 测试样本数
print(n_test) 
# 50

# def plot_kernel_reg(y_hat):
#     d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
#              xlim=[0, 5], ylim=[-1, 5])
#     d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);


# Plotting function using matplotlib
def plot_kernel_reg(y_hat):
    plt.figure(figsize=(8, 4))
    plt.plot(x_test, y_truth, label='Truth', linestyle='-', color='blue')
    plt.plot(x_test, y_hat, label='Pred', linestyle='--', color='orange')
    plt.scatter(x_train, y_train, alpha=0.5, label='Train Data', color='green')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Kernel Regression Baseline')
    plt.xlim([0, 5])
    plt.ylim([-1, 5])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 10.2.2. 平均汇聚
y_train_mean = y_train.mean()
# tensor(2.3274)
y_hat = torch.repeat_interleave(y_train_mean, n_test)
print(y_hat)
# tensor([2.3274, 2.3274, 2.3274, 2.3274, 2.3274, 2.3274, 2.3274, 2.3274, 2.3274,
#         2.3274, 2.3274, 2.3274, 2.3274, 2.3274, 2.3274, 2.3274, 2.3274, 2.3274,
#         2.3274, 2.3274, 2.3274, 2.3274, 2.3274, 2.3274, 2.3274, 2.3274, 2.3274,
#         2.3274, 2.3274, 2.3274, 2.3274, 2.3274, 2.3274, 2.3274, 2.3274, 2.3274,
#         2.3274, 2.3274, 2.3274, 2.3274, 2.3274, 2.3274, 2.3274, 2.3274, 2.3274,
#         2.3274, 2.3274, 2.3274, 2.3274, 2.3274])

plot_kernel_reg(y_hat)


# 10.2.3. 非参数注意力汇聚
# X_repeat的形状:(n_test,n_train),
# 每一行都包含着相同的测试输入（例如：同样的查询）
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
print(X_repeat) # torch.Size([50, 50])
# tensor([[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
#         [0.1000, 0.1000, 0.1000,  ..., 0.1000, 0.1000, 0.1000],
#         [0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000],
#         ...,
#         [4.7000, 4.7000, 4.7000,  ..., 4.7000, 4.7000, 4.7000],
#         [4.8000, 4.8000, 4.8000,  ..., 4.8000, 4.8000, 4.8000],
#         [4.9000, 4.9000, 4.9000,  ..., 4.9000, 4.9000, 4.9000]])

# x_train包含着键。attention_weights的形状：(n_test,n_train),
# 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重, Apply softmax(dim=1) → normalize across each row
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
print(-(X_repeat - x_train)**2 / 2) # torch.Size([50, 50])
# tensor([[-1.5609e-04, -4.9469e-03, -2.4926e-02,  ..., -9.8167e+00,
#          -1.0616e+01, -1.1250e+01],
#         [-3.3892e-03, -1.4171e-07, -7.5984e-03,  ..., -9.3786e+00,
#          -1.0160e+01, -1.0781e+01],
#         [-1.6622e-02, -5.0534e-03, -2.7087e-04,  ..., -8.9505e+00,
#          -9.7146e+00, -1.0322e+01],
#         ...,
#         [-1.0962e+01, -1.0582e+01, -1.0021e+01,  ..., -3.6191e-02,
#          -4.2462e-03, -9.4585e-04],
#         [-1.1435e+01, -1.1048e+01, -1.0473e+01,  ..., -6.8094e-02,
#          -1.8462e-02, -1.5965e-03],
#         [-1.1919e+01, -1.1523e+01, -1.0936e+01,  ..., -1.1000e-01,
#          -4.2677e-02, -1.2247e-02]])
print(attention_weights) # torch.Size([50, 50])
# tensor([[7.6062e-02, 7.5698e-02, 7.4201e-02,  ..., 4.1485e-06, 1.8651e-06,
#          9.8915e-07],
#         [7.0429e-02, 7.0668e-02, 7.0133e-02,  ..., 5.9723e-06, 2.7330e-06,
#          1.4692e-06],
#         [6.4973e-02, 6.5729e-02, 6.6044e-02,  ..., 8.5662e-06, 3.9900e-06,
#          2.1742e-06],
#         ...,
#         [1.0928e-06, 1.5975e-06, 2.8020e-06,  ..., 6.0760e-02, 6.2732e-02,
#          6.2940e-02],
#         [7.3452e-07, 1.0825e-06, 1.9225e-06,  ..., 6.3495e-02, 6.6726e-02,
#          6.7861e-02],
#         [4.9230e-07, 7.3152e-07, 1.3153e-06,  ..., 6.6166e-02, 7.0774e-02,
#          7.2961e-02]])

# y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
y_hat = torch.matmul(attention_weights, y_train)
print(y_hat) # torch.Size([50])
# tensor([1.7075, 1.7737, 1.8412, 1.9097, 1.9788, 2.0482, 2.1174, 2.1858, 2.2529,
#         2.3180, 2.3806, 2.4398, 2.4950, 2.5453, 2.5901, 2.6285, 2.6598, 2.6835,
#         2.6989, 2.7056, 2.7035, 2.6925, 2.6729, 2.6451, 2.6099, 2.5682, 2.5211,
#         2.4698, 2.4154, 2.3593, 2.3024, 2.2459, 2.1906, 2.1370, 2.0859, 2.0375,
#         1.9921, 1.9498, 1.9107, 1.8747, 1.8417, 1.8116, 1.7843, 1.7597, 1.7375,
#         1.7176, 1.6998, 1.6841, 1.6702, 1.6579])

plot_kernel_reg(y_hat)

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(5, 5),
                  cmap='Reds'):
    """Display matrix heatmaps in a grid"""
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    print("num_rows, num_cols", num_rows, num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False,
                             constrained_layout=True)  # <- use this instead of tight_layout
    print("fig, axes", fig, axes)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        print("i, (row_axes, row_matrices)", i, (row_axes, row_matrices))
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            print("j, (ax, matrix))", j, (ax, matrix))
            matrix_numpy = matrix.detach().numpy()
            pcm = ax.imshow(matrix_numpy, cmap=cmap)
            print("matrix_numpy, pcm", matrix_numpy, pcm)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    # Use all axes for colorbar placement
    fig.colorbar(pcm, ax=axes.ravel().tolist(), shrink=0.6)
    print("pcm, ax", pcm, axes.ravel().tolist())
    plt.show()


show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')


# 10.2.4.1. 批量矩阵乘法
X = torch.ones((2, 1, 4))
# tensor([[[1., 1., 1., 1.]],
#         [[1., 1., 1., 1.]]])

Y = torch.ones((2, 4, 6))
# tensor([[[1., 1., 1., 1., 1., 1.],
#          [1., 1., 1., 1., 1., 1.],
#          [1., 1., 1., 1., 1., 1.],
#          [1., 1., 1., 1., 1., 1.]],
#         [[1., 1., 1., 1., 1., 1.],
#          [1., 1., 1., 1., 1., 1.],
#          [1., 1., 1., 1., 1., 1.],
#          [1., 1., 1., 1., 1., 1.]]])

print(torch.bmm(X, Y))
# tensor([[[4., 4., 4., 4., 4., 4.]],
#         [[4., 4., 4., 4., 4., 4.]]])
print(torch.bmm(X, Y).shape) #torch.Size([2, 1, 6])


weights = torch.ones((2, 10)) * 0.1
print("weights", weights, "weights.unsqueeze(1)", weights.unsqueeze(1))
values = torch.arange(20.0).reshape((2, 10))
print("values", values, "values.unsqueeze(-1)", values.unsqueeze(-1))
print(torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1)))

# 10.2.4.2. 定义模型
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
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)


# 10.2.4.3. 训练
# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = x_train.repeat((n_train, 1))
# tensor([[0.0177, 0.0995, 0.2233,  ..., 4.4310, 4.6078, 4.7435],
#         [0.0177, 0.0995, 0.2233,  ..., 4.4310, 4.6078, 4.7435],
#         [0.0177, 0.0995, 0.2233,  ..., 4.4310, 4.6078, 4.7435],
#         ...,
#         [0.0177, 0.0995, 0.2233,  ..., 4.4310, 4.6078, 4.7435],
#         [0.0177, 0.0995, 0.2233,  ..., 4.4310, 4.6078, 4.7435],
#         [0.0177, 0.0995, 0.2233,  ..., 4.4310, 4.6078, 4.7435]])

# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))
# tensor([[-0.0255,  0.1290,  0.3687,  ...,  1.6798,  2.0925,  1.7362],
#         [-0.0255,  0.1290,  0.3687,  ...,  1.6798,  2.0925,  1.7362],
#         [-0.0255,  0.1290,  0.3687,  ...,  1.6798,  2.0925,  1.7362],
#         ...,
#         [-0.0255,  0.1290,  0.3687,  ...,  1.6798,  2.0925,  1.7362],
#         [-0.0255,  0.1290,  0.3687,  ...,  1.6798,  2.0925,  1.7362],
#         [-0.0255,  0.1290,  0.3687,  ...,  1.6798,  2.0925,  1.7362]])

mask = (1 - torch.eye(n_train)).type(torch.bool)
print("mask", mask)
# keys的形状:('n_train'，'n_train'-1)
keys = X_tile[mask].reshape((n_train, -1))
# tensor([[0.0995, 0.2233, 0.4149,  ..., 4.4310, 4.6078, 4.7435],
#         [0.0177, 0.2233, 0.4149,  ..., 4.4310, 4.6078, 4.7435],
#         [0.0177, 0.0995, 0.4149,  ..., 4.4310, 4.6078, 4.7435],
#         ...,
#         [0.0177, 0.0995, 0.2233,  ..., 4.3954, 4.6078, 4.7435],
#         [0.0177, 0.0995, 0.2233,  ..., 4.3954, 4.4310, 4.7435],
#         [0.0177, 0.0995, 0.2233,  ..., 4.3954, 4.4310, 4.6078]])
# torch.Size([50, 49])

# values的形状:('n_train'，'n_train'-1)
values = Y_tile[mask].reshape((n_train, -1))
# tensor([[ 0.1290,  0.3687,  0.7393,  ...,  1.6798,  2.0925,  1.7362],
#         [-0.0255,  0.3687,  0.7393,  ...,  1.6798,  2.0925,  1.7362],
#         [-0.0255,  0.1290,  0.7393,  ...,  1.6798,  2.0925,  1.7362],
#         ...,
#         [-0.0255,  0.1290,  0.3687,  ...,  1.5837,  2.0925,  1.7362],
#         [-0.0255,  0.1290,  0.3687,  ...,  1.5837,  1.6798,  1.7362],
#         [-0.0255,  0.1290,  0.3687,  ...,  1.5837,  1.6798,  2.0925]])
# torch.Size([50, 49])

net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)

losses = []
for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    loss_val = float(l.sum())
    losses.append(loss_val)
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')


# Plot training loss
plt.figure(figsize=(6, 4))
plt.plot(range(1, 6), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.title('NWKernelTraining Loss')
plt.tight_layout()
plt.show()

# keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
keys = x_train.repeat((n_test, 1))
# value的形状:(n_test，n_train)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)

show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')