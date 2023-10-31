import scipy.sparse
from train import *
# 创建一个1x10的CSR格式矩阵
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 非零元素
row_indices = [0,0,0,0,0,0,0,0,0,0]  # 行索引
column_indices = list(range(10))  # 列索引
print(len(data),len(row_indices),len(column_indices))
csr_matrix = scipy.sparse.csr_matrix((data, (row_indices, column_indices)),shape = (1,10))

# 创建一个1x10的CSR格式矩阵
data2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 非零元素
row_indices = [0,0,0,0,0,0,0,0,0,0]  # 行索引
column_indices = list(range(10))  # 列索引

test = []
test.append([1,2,3,4,5,6,7,8,9,10])
csr_matrix2 = scipy.sparse.csr_matrix((data2, (row_indices, column_indices)), shape=(1, 10))

recall = Ktop_single(csr_matrix,csr_matrix2,1,10,2,test,0)

print(csr_matrix)
print(csr_matrix2)

