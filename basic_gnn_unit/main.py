import os
import pickle
import logging

from torch import optim

from Read_data_func import read_data
from Basic_Bert_Unit_model import Basic_Bert_Unit_model
from Batch_TrainData_Generator import Batch_TrainData_Generator
from Model import *
from train_func import train

logging.basicConfig(level=logging.ERROR)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from Param import *
from utils import *
import networkx as nx

def compute_adj(rel_triples, entid):
    edge_index = []

    for h, r, t in rel_triples:
        edge_index.append([entid.index(h), entid.index(t)])
    
    # 将边列表转换为无向图
    G = nx.Graph()
    G.add_edges_from(edge_index)
    
    # 将无向图转换为邻接矩阵
    adj_matrix = nx.adjacency_matrix(G)
    
    # 将稀疏矩阵转换为密集矩阵
    dense_matrix = adj_matrix.toarray()
    print(dense_matrix)

    return dense_matrix


def main():

    # read data
    ent_ill, train_ill, test_ill, \
    index2rel, index2entity, rel2index, entity2index, \
    ent2data, rel_triples_1, rel_triples_2, \
    entid_1, entid_2 = read_data()

    print("----------------get entity embedding--------------------")
    cuda_num = CUDA_NUM
    batch_size = 256
    print("GPU NUM:", cuda_num)

    # load basic bert unit model
    bert_model_path = BASIC_BERT_UNIT_MODEL_SAVE_PATH + BASIC_BERT_UNIT_MODEL_SAVE_PREFIX + "model_epoch_" \
                      + str(LOAD_BASIC_BERT_UNIT_MODEL_EPOCH_NUM) + '.p'
    Model = Basic_Bert_Unit_model(768, BASIC_BERT_UNIT_MODEL_OUTPUT_DIM)
    Model.load_state_dict(torch.load(bert_model_path, map_location='cpu'))
    print("loading basic bert unit model from:  {}".format(bert_model_path))
    Model.eval()
    for name, v in Model.named_parameters():
        v.requires_grad = False
    Model = Model.cuda(cuda_num)

    # generate entity embedding by basic bert unit
    start_time = time.time()
    ent_emb = []
    for eid in range(0, len(ent2data.keys()), batch_size):  # eid == [0,n)
        token_inputs = []
        mask_inputs = []
        for i in range(eid, min(eid + batch_size, len(ent2data.keys()))):
            token_input = ent2data[i][0]
            mask_input = ent2data[i][1]
            token_inputs.append(token_input)
            mask_inputs.append(mask_input)
        vec = Model(torch.LongTensor(token_inputs).cuda(cuda_num),
                    torch.FloatTensor(mask_inputs).cuda(cuda_num))
        ent_emb.extend(vec.detach().cpu().tolist())
    print("get entity embedding using time {:.3f}".format(time.time() - start_time))
    print("entity embedding shape: ", np.array(ent_emb).shape)


    print("----------------get gcn-align feature--------------------")


    ent1 = [e1 for e1, e2 in ent_ill]
    ent2 = [e2 for e1, e2 in ent_ill]


    ent_f_1 = []
    ent_f_2 = []
    for i in range(len(entid_1)):
        ent_f_1.append(ent_emb[entid_1[i]])
    for i in range(len(entid_2)):
        ent_f_2.append(ent_emb[entid_2[i]])


    edge_index_1, edge_index_2 = neigh_ent_dict_gene(rel_triples_1, rel_triples_2, entid_1, entid_2)
    # edge_index_1 = compute_adj(rel_triples_1, entid_1)
    # edge_index_2 = compute_adj(rel_triples_2, entid_2)

    Train_gene = Batch_TrainData_Generator(train_ill, ent1, ent2, index2entity, batch_size=TRAIN_BATCH_SIZE, neg_num=NEG_NUM)
    GCN_Model = GCN(300, 512, 300).cuda(cuda_num)
    Optimizer = optim.Adam(GCN_Model.parameters(), lr=LEARNING_RATE)
    Criterion = nn.MarginRankingLoss(margin=MARGIN, size_average=True)
    #generate description/name-view interaction feature


    #train
    train(Model, GCN_Model, Criterion, Optimizer, Train_gene, train_ill, test_ill, ent2data,
          ent_f_1, ent_f_2, edge_index_1, edge_index_2, entid_1, entid_2)






if __name__ == '__main__':
    print(torch.__version__)
    fixed(SEED_NUM)
    main()