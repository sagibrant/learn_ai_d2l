import torch
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
# with open(data_file, 'w') as f:
#     f.write('NumRooms,Alley,Price\n')  # 列名
#     f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')

# 如果没有安装pandas，只需取消对以下行的注释来安装pandas
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print("read data:\r\n", data)


# 1.删除缺失值最多的列。
print("columns",  data.columns)
print("columns by list",  list(data))
for column in data.columns:
    print(column, data[column])

df_count = data.isna().sum()
label = df_count.idxmax()
new_data = data.drop(label, axis=1)
print("删除缺失值最多的列:\r\n", new_data)
# new_data = data.dropna(axis=1, thresh=(data.count().min()+1))

# 2.将预处理后的数据集转换为张量格式。
new_data_tensor = torch.tensor(new_data.to_numpy(dtype=float))
print("将预处理后的数据集转换为张量格式:\r\n", new_data_tensor)
