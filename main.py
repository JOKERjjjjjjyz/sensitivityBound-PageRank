import time
import random
from laplace2 import *
from sensi import *
import dataloader
import world
import torch
from dataloader import Loader
import sys
import scipy.sparse as sp
from train import *
import numpy as np
from scipy.sparse import csr_matrix
import torch.sparse
from scipy.sparse import save_npz

if __name__ == '__main__':
    if world.dataset in ['gowalla', 'yelp2018', 'amazon-book','ml-1m','BlogCatalog']:
        dataset = dataloader.Loader(path="./data/"+world.dataset)
    elif world.dataset == 'lastfm':
        dataset = dataloader.Loader(path="./data")
    #K = 4
    xi = 0.000335
    sigma = 0.000001
    k = 100
    epsilon = 7.0
    with open(f"{world.dataset}_AppPG_top{k}_recall_{epsilon}.txt", 'a') as file:
        # 在需要时写入内容
        file.write(f"This is {world.dataset}_AppPG_top{k}_recall:\n")
    #for ii in range(1,5):
    #    alpha = 0.2*ii
    adversary_list = []
    M = max(dataset.n_user,dataset.m_item)
    for _ in range(1000):
        adversary = random.randint(1, M - 1)
        adversary_list.append(adversary)
    alpha = 0.08
    recall = 0
    precision = 0
    F = 0
    NDCG = 0
    with open(f"{world.dataset}_AppPG_top{k}_recall_{epsilon}.txt", 'a') as file:
        file.write(f"alpha={alpha},epsilon={epsilon} :\n")
    graph,norm_graph = dataset.getSparseGraph()
    C= graph.copy()
    normC = norm_graph.copy()
    sensitivity = sigma
    testarray = [[] for _ in range(M)]
    uservector = dataset.UserItemNet.copy()
    for idx, user in enumerate(dataset.test):
        testarray[idx] = dataset.test[user]
    for s in adversary_list:
        ppr = pushflowcap_vector(C,s,alpha,xi,sigma,"joint",M,normC)
        ppr_item = ppr.copy().reshape(-1)
        dimension = len(ppr_item)
        noise_vector = generate_laplace_noise(dimension, sensitivity/epsilon).reshape(-1)
        dp_ppr = ppr_item.copy() + noise_vector.copy()
        print(dp_ppr.shape)
        re,pre,F1,ndcg = Ktop_single(uservector.getrow(s), dp_ppr, M, 1, 100,testarray,s)
        recall += re
        precision += pre
        F += F1
        NDCG += ndcg
    recall = recall / 1000.
    precision /= 1000.
    F /= 1000.
    NDCG /= 1000.
    with open(f"{world.dataset}_AppPG_top{k}_recall_{epsilon}.txt", 'a') as file:
        file.write(f"top{100} ver:  recall: {recall} pre:{precision} F:{F} ndcg:{NDCG}\n")
        #save_npz(f"{world.dataset}_result_{K}layer_{alpha}_new.npz", rowM(vector_propagate,M))
