from transformers import BertModel, RobertaModel
import torch
import torch.nn as nn


class Basic_Bert_Unit_model(nn.Module):
    def __init__(self,input_size,result_size):
        super(Basic_Bert_Unit_model,self).__init__()
        self.result_size = result_size
        self.input_size = input_size
        self.bert_model = BertModel.from_pretrained('pubmed-bert-abstract')
        # self.bert_model = BertModel.from_pretrained('dmis-lab/biobert-v1.1')
        self.out_linear_layer = nn.Linear(self.input_size,self.result_size)
        self.dropout = nn.Dropout(p = 0.1)



    def forward(self,batch_word_list,attention_mask):
        # token_type_ids =token_type_ids
        x = self.bert_model(input_ids=batch_word_list, attention_mask=attention_mask, return_dict=False)
        sequence_output, pooled_output = x
        cls_vec = sequence_output[:,0]
        output = self.dropout(cls_vec)
        output = self.out_linear_layer(output)
        
        # output_tensor_list = []
        # for batch_id in range(batch_word_list.shape[0]):
        #     # 获取当前样本的input_ids
        #     input_ids = batch_word_list[batch_id]
        #     print(input_ids)
        #     # 使用torch.where函数获取该样本中所有SEP的位置
        #     sep_positions = torch.where(input_ids == 102)[0]
        #     # 根据需要的SEP位置获取当前样本的张量
        #     start_position = sep_positions[0] + 1  # 第一个SEP后面的位置
        #     if len(sep_positions) >= 2:
        #         end_position = sep_positions[1]        # 第四个SEP的位置
        #         tensor_list = sequence_output[batch_id, start_position:end_position, :]
        #     else:
        #         tensor_list = sequence_output[batch_id, start_position:, :]
        #     output_tensor_list.append(tensor_list)
            
        # # 将每个样本的张量组合成一个张量列表
        # if len(output_tensor_list) > 0:
        #     sep_output = torch.cat(output_tensor_list, dim=0)
        # else:
        #     sep_output = torch.empty(0)
        
        # sep_output = self.dropout(sep_output)
        # sep_output = self.out_linear_layer(sep_output)

        return output

