import numpy as np
from scipy.stats import gamma
def sample_vector_gamma(d, b):
    a = np.random.gamma(shape=d, scale=1/b)
    cdf_value = gamma.cdf(a, d, 0,scale=1/b)
    print(f"p(a={a})={cdf_value}")
    u = np.random.normal(0, 1, d)
    norm_u = np.linalg.norm(u)
    o = a * u / norm_u
    return o
def generate_laplace_noise(d, b, size=1): 
    laplace_noise = np.random.laplace(loc=0, scale=b, size=(size, d)) 
    return laplace_noise
def testLaplace(d,b,size=1):
    laplace_noise = np.random.laplace(loc=0, scale=b, size=(size, d))
    return laplace_noise
# 输入维度和分布参数
#d = 10  # 维度
#b = 1.0  # 分布参数

# 生成向量 'o' 样
#sampled_o = sample_vector(d, b)
#print("Generated vector 'o':")
#print(sampled_o)

