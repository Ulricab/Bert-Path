from model_train_test_func import *
from Param import *

def main():
    print("----------------interaction model--------------------")
    cuda_num = CUDA_NUM
    print("GPU num {}".format(cuda_num))
    #print("ko~ko~da~yo~")

    # read other data from bert unit model(train ill/test ill/eid2data)
    # (These files were saved during the training of basic bert unit)
    bert_model_other_data_path = BASIC_BERT_UNIT_MODEL_SAVE_PATH + BASIC_BERT_UNIT_MODEL_SAVE_PREFIX + 'other_data.pkl'
    train_ill, test_ill, eid2data, path2data = pickle.load(open(bert_model_other_data_path, "rb"))
    print("train_ill num: {} /test_ill num:{} / train_ill & test_ill num: {}".format(len(train_ill),len(test_ill), len(set(train_ill) & set(test_ill) )))


    #(candidate) entity pairs
    entity_pairs = pickle.load(open(ENT_PAIRS_PATH, "rb"))

    #interaction features
    nei_features = pickle.load(open(NEIGHBORVIEW_SIMILARITY_FEATURE_PATH_1, "rb"))#neighbor-view interaction similarity feature
    des_features = pickle.load(open(DESVIEW_SIMILARITY_FEATURE_PATH, "rb")) #description/name-view interaction similarity feature
    path_features = pickle.load(open(PATH_DESVIEW_SIMILARITY_FEATURE_PATH, "rb"))
    gcn_features = pickle.load(open(GCNVIEW_SIMILARITY_FEATURE_PATH, "rb"))
    # path_features = pickle.load(open(NEIGHBORVIEW_SIMILARITY_FEATURE_PATH_2, "rb"))
    train_candidate = pickle.load(open(TRAIN_CANDIDATES_PATH, "rb"))
    test_candidate = pickle.load(open(TEST_CANDIDATES_PATH, "rb"))
    all_features = [] 
    for i in range(len(entity_pairs)):
        all_features.append(des_features[i] + nei_features[i] + gcn_features[i] + path_features[i]) 
    print("All features embedding shape: ", np.array(all_features).shape)



    entpair2f_idx = {entpair: feature_idx for feature_idx, entpair in enumerate(entity_pairs)}
    Train_gene = Train_index_generator(train_ill, train_candidate, entpair2f_idx,neg_num=NEG_NUM, batch_size=BATCH_SIZE)
    Model = MlP(1 + 42 + 1 + 1, 11).cuda(cuda_num)
    Optimizer = optim.Adam(Model.parameters(), lr=LEARNING_RATE)
    Criterion = nn.MarginRankingLoss(margin=MARGIN, size_average=True)

    #train
    train(Model, Optimizer, Criterion, Train_gene, all_features, test_candidate, test_ill,
          entpair2f_idx, epoch_num=EPOCH_NUM, eval_num=10, cuda_num=cuda_num, test_topk=50)

    #save
    # torch.save(Model.state_dict(), "../Save_model/interaction_model_enen" + '.p')


if __name__ == '__main__':
    fixed(SEED_NUM)
    main()