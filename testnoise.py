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
    if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
        dataset = dataloader.Loader(path="./data/"+world.dataset)
    elif world.dataset == 'lastfm':
        dataset = dataloader.Loader(path="./data")
    K = 2
    #with open(f"{world.dataset}_layer{K}_top20_noise_recall.txt", 'a') as file:
        # 在需要时写入内容
    #    file.write(f"This is {world.dataset}_maxlayer{K}_top20_recall:\n")
    M = dataset.n_users
    N = dataset.m_items
    adversary_list = []
    for _ in range(10):
        adversary = random.randint(1, M - 1)
        adversary_list.append(adversary)

    for ii in range(1,2):
        alpha = 0.2 *ii
        testarray = [[] for _ in range(M)]
        uservector = dataset.UserItemNet
        for idx, user in enumerate(dataset.test):
            testarray[idx] = dataset.test[user]
        vector_propagate_M = sp.load_npz(f"{world.dataset}_result_{K}layer_{alpha}.npz")
        dimension = vector_propagate_M.shape[1]
        recall1 = 0
        for attacker in adversary_list:
            recall1 += Ktop_single(uservector[attacker],vector_propagate_M[attacker],M,N,20,testarray,attacker)
        recall1 = recall1 / len(adversary_list)
        print(f"origin:{recall1}")
        #with open(f"{world.dataset}_layer{K}_top20_noise_recall.txt", 'a') as file:
        #    file.write(f"{K}layer_origin_recall = {recall1}\n")
        sensitivity = jyz_brute(K+1,20,alpha)
        print(f"sensi:{sensitivity}")
        recall = 0
        for adversary in adversary_list:
            noise_vector = generate_laplace_noise(dimension,sensitivity)
            noise_matrix = sp.csr_matrix(noise_vector)
            print(noise_vector)
            #print(vector_propagate_M[adversary])
            #print(noise_matrix + vector_propagate_M[adversary])
            recall += Ktop_single_noise(uservector[adversary], vector_propagate_M[adversary],noise_matrix, M, N, 20,testarray,adversary)
        recall = recall / len(adversary_list)
        print(f"noise:{recall}")
        #with open(f"{world.dataset}_layer{K}_top20_noise_recall.txt", 'a') as file:
        #    file.write(f"{K}layer_noise_recall = {recall}\n") 
