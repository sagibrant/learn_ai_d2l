import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(x + y, x * y, x / y, x**y)

x = torch.arange(4)
print(x, len(x), x.shape)

A = torch.arange(20).reshape(5, 4)
print(A, A.T)

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B, B.T, B == B.T)


X = torch.arange(24).reshape(2, 3, 4)
print(X)

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print(A, A + B, A * B)


a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X, (a * X).shape)

# 降维
x = torch.arange(4, dtype=torch.float32)
print(x, x.sum())

print(A.shape, A.sum())

A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0, A_sum_axis0.shape)

A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1, A_sum_axis1.shape)

print(A.sum(axis=[0, 1]))  # 结果和A.sum()相同

print(A.mean(), A.sum() / A.numel())
# 非降维求和
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)

print(A / sum_A)

print(A.cumsum(axis=0))

# 点积（Dot Product）
y = torch.ones(4, dtype = torch.float32)
print(x, y, torch.dot(x, y))

print(torch.sum(x * y))

# 矩阵-向量积
print(A.shape, x.shape, torch.mv(A, x))

# 矩阵乘法
B = torch.ones(4, 3)
print(torch.mm(A, B))

# 范数 norm

u = torch.tensor([3.0, -4.0])
print(torch.norm(u)) # L2 范数

print(torch.abs(u).sum()) # L1 范数

print(torch.norm(torch.ones((4, 9)))) # Frobenius范数

# 题目8，为linalg.norm函数提供3个或更多轴的张量，并观察其输
print(torch.linalg.norm(torch.ones(4,9,10))) # => tensor(18.9737)
print(torch.linalg.norm(torch.arange(20,dtype=torch.float32))) # => tensor(49.6991)
print(torch.linalg.norm(torch.arange(20,dtype=torch.float32).reshape(5,2,2))) # => 49.6991