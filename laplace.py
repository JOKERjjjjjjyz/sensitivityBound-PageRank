import numpy as np

def generate_multivariate_laplace_noise(dimension, b, size=1):
        # 生成多维拉普拉斯噪声，dimension是维度，size是生成的样本数
        laplace_noise = np.random.laplace(scale=b, size=(size, dimension))
        return laplace_noise

# 向量的维度
N = 10

# 尺度参数
b = 1.0

# 生成 N 维拉普拉斯噪声向量
noise_vector = generate_multivariate_laplace_noise(N, b)

print("生成的拉普拉斯噪声向量:")
print(noise_vector)
