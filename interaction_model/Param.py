"""
hyper-parameters:
"""
CUDA_NUM = 0 #GPU num
CLASS = 'icd10_icd11' #class
ENTITY_NEIGH_MAX_NUM = 50 # max sampling neighbor num of entity
ENTITY_ATTVALUE_MAX_NUM = 50 #max sampling attributeValue num of entity
KERNEL_NUM = 21
SEED_NUM = 11037
CANDIDATE_NUM = 50 # candidate number

BATCH_SIZE = 64 # train batch size
NEG_NUM = 5 # negative sampling num
LEARNING_RATE = 6e-4 # learning rate
MARGIN = 1 # margin
EPOCH_NUM = 60# train epoch num

INTERACTION_MODEL_SAVE_PATH = "../Save_model/interaction_model_{}.bin".format(CLASS) #interaction model save path.

#load model(base_bert_unit_model) path
BASIC_BERT_UNIT_MODEL_SAVE_PATH = "../Save_model/"
BASIC_BERT_UNIT_MODEL_SAVE_PREFIX = "MEDICAL_{}".format(CLASS)
LOAD_BASIC_BERT_UNIT_MODEL_EPOCH_NUM = 4
BASIC_BERT_UNIT_MODEL_OUTPUT_DIM = 300

GCN_MODEL_SAVE_PATH = "../Save_model/"
GCN_MODEL_SAVE_PREFIX = "gcn_model_{}".format(CLASS)

#load data path
DATA_PATH = r"../data/medical_base/{}/".format(CLASS)


#candidata_save_path
TRAIN_CANDIDATES_PATH = DATA_PATH + 'train_candidates.pkl'
TEST_CANDIDATES_PATH = DATA_PATH + 'test_candidates.pkl'

#entity embedding and attributeValue embedding save path.
ENT_EMB_PATH = DATA_PATH + '{}_emb_{}.pkl'.format(BASIC_BERT_UNIT_MODEL_SAVE_PREFIX,LOAD_BASIC_BERT_UNIT_MODEL_EPOCH_NUM)
ENT_GCN_EMB_PATH = DATA_PATH + '{}_emb_{}.pkl'.format(GCN_MODEL_SAVE_PREFIX,'final')
PATH_EMB_PATH = DATA_PATH + '{}_path_emb_{}.pkl'.format(BASIC_BERT_UNIT_MODEL_SAVE_PREFIX,LOAD_BASIC_BERT_UNIT_MODEL_EPOCH_NUM)
PATH_SEP_EMB_PATH = DATA_PATH + '{}_path_sep_emb_{}.pkl'.format(BASIC_BERT_UNIT_MODEL_SAVE_PREFIX,LOAD_BASIC_BERT_UNIT_MODEL_EPOCH_NUM)
ATTRIBUTEVALUE_EMB_PATH = DATA_PATH + 'attributeValue_embedding.pkl'
ATTRIBUTEVALUE_LIST_PATH = DATA_PATH + 'attributeValue_list.pkl' #1-1 match to attributeValue embedding.

#(candidate) entity_pairs save path.
ENT_PAIRS_PATH = DATA_PATH + 'ent_pairs.pkl' #[(e1,ea),(e1,eb)...]

#interaction feature save filepath name
NEIGHBORVIEW_SIMILARITY_FEATURE_PATH_1 = DATA_PATH + 'neighbor_view_similarity_feature_1.pkl' #1-1 match to entity_pairs
NEIGHBORVIEW_SIMILARITY_FEATURE_PATH_2 = DATA_PATH + 'neighbor_view_similarity_feature_2.pkl'
DESVIEW_SIMILARITY_FEATURE_PATH = DATA_PATH + 'des_view_similarity_feature.pkl' #1-1 match to entity_pairs
PATH_DESVIEW_SIMILARITY_FEATURE_PATH = DATA_PATH + 'path_des_view_similarity_feature.pkl' #1-1 match to entity_pairs
PATH_SEP_SIMILARITY_FEATURE_PATH = DATA_PATH + 'path_des_view_similarity_feature.pkl' #1-1 match to entity_pairs
GCNVIEW_SIMILARITY_FEATURE_PATH = DATA_PATH + 'gcn_view_similarity_feature.pkl' #1-1 match to entity_gcn_pairs
