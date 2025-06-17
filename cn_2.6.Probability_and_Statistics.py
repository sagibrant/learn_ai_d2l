# %matplotlib inline
import matplotlib.pyplot as plt
import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([6]) / 6 # tensor([0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667])
counts = multinomial.Multinomial(1, fair_probs).sample() # tensor([0., 0., 1., 0., 0., 0.])
print(counts)

counts = multinomial.Multinomial(10, fair_probs).sample() # tensor([2., 0., 2., 3., 0., 3.])
print(counts)

# 将结果存储为32位浮点数以进行除法
counts = multinomial.Multinomial(1000, fair_probs).sample()  # tensor([155., 164., 165., 162., 182., 172.])
print(counts / 1000) # 相对频率作为估计值  tensor([0.1550, 0.1640, 0.1650, 0.1620, 0.1820, 0.1720])

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
print(counts)
# tensor([[1., 1., 1., 4., 1., 2.],
#         [1., 1., 3., 2., 3., 0.],
#         [0., 1., 3., 2., 3., 1.],
#         ...,
#         [2., 2., 1., 4., 0., 1.],
#         [1., 0., 1., 5., 1., 2.],
#         [3., 3., 0., 2., 0., 2.]])

cum_counts = counts.cumsum(dim=0)
print(cum_counts)
# tensor([[  1.,   1.,   1.,   4.,   1.,   2.],
#         [  2.,   2.,   4.,   6.,   4.,   2.],
#         [  2.,   3.,   7.,   8.,   7.,   3.],
#         ...,
#         [811., 828., 879., 855., 791., 816.],
#         [812., 828., 880., 860., 792., 818.],
#         [815., 831., 880., 862., 792., 820.]])

estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
print(estimates)
# tensor([[0.1000, 0.1000, 0.1000, 0.4000, 0.1000, 0.2000],
#         [0.1000, 0.1000, 0.2000, 0.3000, 0.2000, 0.1000],
#         [0.0667, 0.1000, 0.2333, 0.2667, 0.2333, 0.1000],
#         ...,
#         [0.1629, 0.1663, 0.1765, 0.1717, 0.1588, 0.1639],
#         [0.1627, 0.1659, 0.1764, 0.1723, 0.1587, 0.1639],
#         [0.1630, 0.1662, 0.1760, 0.1724, 0.1584, 0.1640]])

# --- Plotting with matplotlib ---
plt.figure(figsize=(6, 4.5))
for i in range(6):
    plt.plot(estimates[:, i].numpy(), label=f"P(die={i + 1})")

plt.axhline(y=1/6, color='black', linestyle='dashed')
plt.xlabel("Groups of experiments")
plt.ylabel("Estimated probability")
plt.legend()
plt.title("Estimated Probabilities of Die Faces Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()