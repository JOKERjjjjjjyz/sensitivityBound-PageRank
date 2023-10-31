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
    epsilon = world.config['lr']
    print(epsilon)
    with open(f"{world.dataset}_layer{K}_top20_Laplace{epsilon}_recall3.txt", 'a') as file:
        # 在需要时写入内容
        file.write(f"This is {world.dataset}_maxlayer{K}_top20_recall:\n")
    graph,norm_graph = dataset.getSparseGraph()
    M = dataset.n_users
    N = dataset.m_items
    adversary_list = []
    for _ in range(100):
        adversary = random.randint(1, M - 1)
        adversary_list.append(adversary)
    for ii in range(1,5):
        recall1 = [0,0,0,0,0,0,0,0,0,0]
        recall = [0,0,0,0,0,0,0,0,0,0]
        alpha = 0.8 + 0.04 *ii
        with open(f"{world.dataset}_layer{K}_top20_Laplace{epsilon}_recall3.txt", 'a') as file:
            file.write(f"alpha={alpha} :\n")
        # K_value = eval(world.topks)
        # K = K_value[0]
        # alpha = world.config['lr']
        testarray = [[] for _ in range(M)]
        uservector = dataset.UserItemNet
        for idx, user in enumerate(dataset.test):
            testarray[idx] = dataset.test[user]
        vector_propagate_M = sp.load_npz(f"{world.dataset}_result_{K}layer_{alpha}_new.npz")
        dimension = vector_propagate_M.shape[1]
        #print(dimension)
        #print(vector_propagate_M.shape)
        for j in range(0,10):
            for attacker in adversary_list:
                recall1[j] += Ktop_single(uservector[attacker],vector_propagate_M[attacker],M,N,20,testarray,attacker)
            recall1[j] = recall1[j] / len(adversary_list)
        #with open(f"{world.dataset}_layer{K}_top20_noise_recall.txt", 'a') as file:
        #    file.write(f"{K}layer_origin_recall = {recall1}\n")
            sensitivity = jyz_brute(K+1,20,alpha)
            for adversary in adversary_list:
                noise_vector = generate_laplace_noise(dimension, sensitivity/epsilon)
                noise_matrix = sp.csr_matrix(noise_vector)
                #print(type(noise_vector),type(vector_propagate_M))
                recall[j] += Ktop_single(uservector[adversary], vector_propagate_M[adversary]+noise_matrix, M, N, 20,testarray,adversary)
            recall[j] = recall[j] / len(adversary_list)
        re1 = sum(recall1)/len(recall1)
        re2 = sum(recall)/len(recall)
        with open(f"{world.dataset}_layer{K}_top20_Laplace{epsilon}_recall3.txt", 'a') as file:
            file.write(f"{K}layer_origin_avg_recall = {re1}\n{K}layer_noise_avg_recall = {re2}\n") 
