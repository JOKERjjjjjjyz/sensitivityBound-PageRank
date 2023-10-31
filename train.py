import numpy as np
from multiprocessing import Pool
import math
import scipy.sparse as sp
from scipy.sparse import csr_matrix
def rowM(matrix,M):
    B = matrix[:M]
    return B
def Mrow(matrix,M):
    B = matrix[:M]
    B = B.transpose()
    B = matrix[:M]
    B = B.transpose()
    print (B.shape)
    return B

def Ktop_single(vector_origin,vector_propagate,M,N,k,test,adversary):
    count = 0
    test_matrix = np.zeros((M,k))
    r_matrix = np.zeros((M,k))
    vector = vector_propagate.copy()
    #print(vector.shape)
    topk_indices = np.argsort(vector, axis=0)[-k:]
    print(adversary,test[adversary],topk_indices)
    recall_count=0
    test2 = np.array(test[adversary])
    for index, item in enumerate(topk_indices):
        if item in test2:
            recall_count += 1
            r_matrix[adversary, index] = 1
    print(recall_count)
    if (len(test[adversary]) == 0):
        print(adversary)
        exit
    recall = recall_count / float(max(1,len(test[adversary])))
    precision = recall_count / float(k)
    length = k if k <= len(test[adversary]) else len(test[adversary])
    test_matrix[adversary, :length] = 1
    max_r = test_matrix.copy()
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = np.sum(r_matrix * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    ndcg_value = np.sum(ndcg)
    F1 = 2 * recall * precision / (1+recall + precision)
    print(f"adversary{adversary}_recall:{recall}")
    return recall,precision,F1,ndcg_value
def Ktop_single_noise(vector_origin,vector_propagate,noise,M,N,k,test,adversary):
    recall = 0
    count = 0
    vector = vector_propagate + noise - 100 * vector_origin
    vector_array = vector.toarray()
    noise_array = noise.toarray()
    pro_array = vector_propagate.toarray()
    topk_indices = np.argsort(vector_array, axis=1)[:, -k:]
    for item in topk_indices:
        print(vector_array[0,item],noise_array[0,item],pro_array[0,item])
    print(adversary,test[adversary],topk_indices)
    for item in test[adversary]:
        if item in topk_indices: recall +=1
    recall = recall / len(test[adversary])
    print(f"adversary{adversary}_recall:{recall}")
    return recall
def Ktop(vector_origin,vector_propagate,M,N,k,test):
    recall = 0
    count = 0
    recall_count = 0
    precision = 0
    F1 = 0
    ndcg_value = 0
    length = 0
    test_matrix = np.zeros((M,k))
    r_matrix = np.zeros((M,k))
    vector = vector_propagate - 100 * vector_origin
    vector_array = vector.toarray()
    topk_indices = np.argsort(vector_array, axis=1)[:, -k:]
    print (topk_indices,type(topk_indices),topk_indices.shape)
    for user in range(M):
        recall_count = 0
        for index, item in enumerate(topk_indices[user]):
            count+=1
            #print("Ktop:count")
            if item in test[user]:
                recall_count +=1
                r_matrix[user,index] = 1
        if (len(test[user]) == 0):
            print(user)
            exit
        recall += recall_count / float(len(test[user]))
        precision += recall_count / float(k)
        length = k if k<=len(test[user]) else len(test[user])
        test_matrix[user,:length] = 1
    max_r = test_matrix.copy()
    idcg = np.sum(max_r * 1./np.log2(np.arange(2,k+2)), axis=1)
    dcg = np.sum(r_matrix * 1./np.log2(np.arange(2,k+2)), axis=1)
    #idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 1.
    ndcg_value = np.sum(ndcg) / float(M)
    recall /= float(M)
    precision /= float(M)
    F1 = 2 * recall * precision / (recall + precision)
    return recall,precision,F1,ndcg_value
def topK(vector, M, N, k, user_start, user_end):
    recommendList = []
    # recommend_vector = [np.zeros(N) for _ in range(M)]
    for user in range(user_start, user_end):
        dense = vector[user].toarray()
        dense_vector_user = dense[0]
        # print("user",user,":",dense_vector_user)
        sorted_indices = np.argsort(dense_vector_user)
        # print("user", user, ":", sorted_indices)
        topk_indices = sorted_indices[-k:]
        # print("user", user, ":", topk_indices)
        for idx in topk_indices:
            # print(user,idx)
            # recommend_vector[user][idx] = 1
            recommendList.append((user, idx))
    return recommendList

def parallel_topK(vector_origin, vector_propagate, M, N, k, num_cores):
    chunk_size = M // num_cores
    vector = vector_propagate - vector_origin
    # print(vector)
    pool = Pool(num_cores)
    results = []
    for i in range(num_cores):
        user_start = i * chunk_size
        user_end = (i + 1) * chunk_size if i < num_cores - 1 else M
        results.append(pool.apply_async(topK, (vector, M, N, k, user_start, user_end)))
    pool.close()
    pool.join()

    recommendList = []
    recommend_vector = [np.zeros(N) for _ in range(M)]
    for result in results:
        partial_recommendList = result.get()
        recommendList.extend(partial_recommendList)
        # for user in range(M):
        #     recommend_vector[user] += partial_recommend_vector[user]
    return recommendList

def pushflowcap(G, s, alpha, xi, sigma, type,V):
    '''
    :param G: <csr matrix>:adjacent matrix D^-1*A
    :param s: source node ID
    :param alpha: lazy walk ratio (different from normal def)
    :param xi: precision for Appro-PG
    :param sigma: sensitivity bound
    :param type: non-joint/ joint
    :param V: node numbers
    :return: s's ppr vector
    '''
    # G is a csr matrix
    # E = np.nonzero(G)  # 邻接矩阵的非零元素对应于边
    # 步骤 3: 初始化参数
    S = set([s])  # 初始时，S 只包含源节点 s
    print(S)
    d = np.sum(G, axis=1)  # 度向量
    p = np.zeros(V)  # PPR 初始值
    r = np.zeros(V)  # residual 初始值
    r[s] = 1
    h = np.zeros(V)  # total pushed flow 初始值
    f = np.zeros(V)
    print(d)
    print(r)
    R = int(math.log(1/xi) / alpha)  # rounds
    T = np.full(V, sigma / (2 * (2 - alpha)))  # thresholds
    print(R,T)
    if type == "joint":
        T[s] = 999
    print(T)
    for i in range(1, R + 1):
        print(i)
        for v in S:
            f[v] = np.minimum(r[v], d[v] * T[v] - h[v])  # Compute the flow to push for node v
            h[v] = h[v] + f[v]  # Update the total pushed flow of node v
            p[v] = p[v]
            r[v] = r[v] - f[v]

        S_prime = S.copy()

        for v in S:
            #print(i,"::",v)
            if f[v] > 0:
                p[v] = p[v] + alpha * f[v]
                r[v] = r[v] + (1 - alpha) / 2 * f[v]

                start_idx = G.indptr[v]
                end_idx = G.indptr[v + 1]
                u_indices = G.indices[start_idx:end_idx]
                for u in u_indices: #??这句的语法可能有问题
                    #print(v,":",u)
                    r[u] = r[u] + (1 - alpha) * f[v] / (2 * d[v])
                    S_prime.add(u)

        S = S_prime

    return p

def pushflowcap_vector(G, s, alpha, xi, sigma, type1,V,normG):
    '''
    :param G: <csr matrix>:adjacent matrix D^-1*A
    :param s: source node ID
    :param alpha: lazy walk ratio (different from normal def)
    :param xi: precision for Appro-PG
    :param sigma: sensitivity bound
    :param type: non-joint/ joint
    :param V: node numbers
    :return: s's ppr vector
    '''
    S = set([s])  # 初始时，S 只包含源节点 s
    s_vector = np.zeros(V)
    s_vector[s] = 1
    d = np.sum(G, axis=1).A  # 度向量
    d = np.squeeze(d)
    p = np.zeros(V)  # PPR 初始值
    r = np.zeros(V)  # residual 初始值
    r[s] = 1
    h = np.zeros(V)  # total pushed flow 初始值
    f = np.zeros(V)
    R = int(math.log(1/xi) / alpha)  # rounds
    print(R,f.shape)
    T = np.full(V, sigma / (2 * (2 - alpha)))  # thresholds
    if type1 == "joint":
        T[s] = 999

    for i in range(1, R + 1):
        #print((d*T).shape,h.shape)
        f = np.minimum(r, d * T - h)  # Compute the flow to push for node v
        #print(f.shape)
        #print("threshold")
        f = f * s_vector
        h = h + f  # Update the total pushed flow of node v
        r = r - f
        #print("S_prime")
        S_prime = S.copy()

        p = p + alpha * f
        r = r + (1 - alpha) / 2 * f
        #print("normG start")
        #print(type(normG),normG.shape)
        #print(type(f),f.shape)
        f_sparse = csr_matrix(f)
        result = f_sparse.dot(normG)
        result_array = result.toarray()
        r = r + (1 - alpha) / 2 * result_array
        #print("normG_end")
        #print("neighbour start")
        S = UpdateNeighbour(G,S_prime)
        #print("neighbour finish")
        for node in S:
            s_vector[node] = 1
    return p

def UpdateNeighbour(G,S_prime):
    S = set(S_prime)
    for v in S_prime:
        S |= (Neighbour(G,v))
    return  S

def Neighbour(G,v):
    S = set([v])
    start_idx = G.indptr[v]
    end_idx = G.indptr[v + 1]
    u_indices = G.indices[start_idx:end_idx]
    for u in u_indices:  
        S.add(u)
    return S


