from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import pickle
import time
from Param import *
from utils import fixed,cos_sim_mat_generate,batch_topk
from GNN_Model import *
from read_data_func import read_structure_datas, read_structure_datas_gcn
from utils import *


def test_read_emb(ent_emb,train_ill,test_ill, bs = 128,candidate_topk = 50):
    test_ids_1 = [e1 for e1, e2 in test_ill]
    test_ids_2 = [e2 for e1, e2 in test_ill]
    test_emb1 = np.array(ent_emb)[test_ids_1].tolist()
    test_emb2 = np.array(ent_emb)[test_ids_2].tolist()
    train_ids_1 = [e1 for e1, e2 in train_ill]
    train_ids_2 = [e2 for e1, e2 in train_ill]
    train_emb1 = np.array(ent_emb)[train_ids_1].tolist()
    train_emb2 = np.array(ent_emb)[train_ids_2].tolist()

    print("Eval entity emb sim in train set.")
    emb1 = train_emb1
    emb2 = train_emb2

    res_mat = cos_sim_mat_generate(emb1, emb2, bs)
    score, index = batch_topk(res_mat, bs, candidate_topk, largest=True)
    test_topk_res(index)
    
    print("Eval entity emb sim in test set.")
    emb1 = test_emb1
    emb2 = test_emb2

    res_mat = cos_sim_mat_generate(emb1, emb2, bs)
    score, index = batch_topk(res_mat, bs, candidate_topk, largest=True)
    test_topk_res(index)

def get_edge_index(rel_triples_1, rel_triples_2, entid_1, entid_2):
    """
    get one hop neighbor of entity
    return a dict, key = entity, value = (padding) neighbors of entity
    """
    edge_index_1 = [[], []]
    edge_index_2 = [[], []]

    for h, r, t in rel_triples_1:
        edge_index_1[0].append(entid_1.index(h))
        edge_index_1[1].append(entid_1.index(t))

    for h, r, t in rel_triples_2:
        edge_index_2[0].append(entid_2.index(h))
        edge_index_2[1].append(entid_2.index(t))

    return edge_index_1, edge_index_2


def main():
    print("----------------get gcn embedding--------------------")
    cuda_num = CUDA_NUM
    print("GPU num: {}".format(cuda_num))

    # read structure data
    ent_ill, index2rel, index2entity, \
    rel2index, entity2index, \
    rel_triples_1, rel_triples_2, \
    entid_1, entid_2 = read_structure_datas_gcn(DATA_PATH)
    rel_triples = []#all relation triples
    rel_triples.extend(rel_triples_1)
    rel_triples.extend(rel_triples_2)

    GCN_model_path = GCN_MODEL_SAVE_PATH + GCN_MODEL_SAVE_PREFIX + '.p'
    GCN_Model = GCN(300, 512, 300)
    GCN_Model.load_state_dict(torch.load(GCN_model_path, map_location='cpu'))
    print("loading basic GCN model from:  {}".format(GCN_model_path))
    GCN_Model.eval()
    for name, v in GCN_Model.named_parameters():
        v.requires_grad = False
    GCN_Model = GCN_Model.cuda(cuda_num)

    # load entity embedding
    ent_emb = pickle.load(open(ENT_EMB_PATH,"rb"))
    print("read entity embedding shape:", np.array(ent_emb).shape)

    # read other data from bert unit model(train ill/test ill/eid2data)
    # (These files were saved during the training of basic bert unit)
    bert_model_other_data_path = BASIC_BERT_UNIT_MODEL_SAVE_PATH + BASIC_BERT_UNIT_MODEL_SAVE_PREFIX + 'other_data.pkl'

    ent_f_1 = []
    ent_f_2 = []
    for i in range(len(entid_1)):
        ent_f_1.append(ent_emb[entid_1[i]])
    for i in range(len(entid_2)):
        ent_f_2.append(ent_emb[entid_2[i]])

    edge_index_1, edge_index_2 = get_edge_index(rel_triples_1, rel_triples_2, entid_1, entid_2)

    new_f_1 = GCN_Model(torch.FloatTensor(ent_f_1).cuda(CUDA_NUM), 
        torch.LongTensor(edge_index_1).cuda(CUDA_NUM))
    
    new_f_2 = GCN_Model(torch.FloatTensor(ent_f_2).cuda(CUDA_NUM), 
        torch.LongTensor(edge_index_2).cuda(CUDA_NUM))
    
    for i in range(len(entid_1)):
        ent_emb[entid_1[i]] = new_f_1[i].detach().cpu().tolist()

    for i in range(len(entid_2)):
        ent_emb[entid_2[i]] = new_f_2[i].detach().cpu().tolist()


    #save gcn embedding.
    pickle.dump(ent_emb, open(ENT_GCN_EMB_PATH, "wb"))
    print("save entity gcn embedding....")







if __name__ == '__main__':
    fixed(SEED_NUM)
    main()







