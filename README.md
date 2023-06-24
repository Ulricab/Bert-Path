# Bert-Path
the source code of paper "Integration of Multiple Terminology Bases Integration of Multiple Terminology Bases: A Multi-View Alignment Method Using The Hierarchical Structure"

Further improvements will be made after publication.

### Dependencies

- Python 3
- Pytorch
- transformers
- Numpy
- torch_geometric
- PubMedBert-abstract(https://www.huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)

### moudles description

The "basic_bert_unit" and "basic_gnn_unit" modules form the foundation for subsequent models. The "interaction_model" module contains the majority of the feature calculation code. The "data" module contains the dataset, while the "get_her_path" module includes the path data processing code.

### How to Run

The model runs in two steps:

#### 1. Fine-tune Basic BERT Unit

To fine-tune the Basic BERT Unit, use: 

```shell
cd basic_bert_unit/
python main.py
```

#### 2. Fine-tune Basic GNN Unit
To fine-tune the Basic GNN Unit, use: 

```shell
cd basic_gnn_unit/
python main.py
```
#### 3. Run the interaction model
```shell
cd ../interaction_model/
bash run.sh
```

### Dataset

- `data/medical_base/icd10_icd11`: We provide a processed dataset of ICD-10 and ICD-11, formatted in the same manner as the standard DBP15K dataset. The "path" files in this dataset were obtained through the code in the "get_her_path" directory. This dataset has been fully processed and is ready for use in experimentation.

Note: Due to copyright reasons, we may not be able to provide data related to SNOMED-CT. However, we can provide the respective websites for the original files, as follows:
- ICD-10: https://icd.who.int/browse10/2019/en
- ICD-11: https://icd.who.int/browse11/l-m/en 
- SNOMED-CT: https://www.nlm.nih.gov/research/umls/mapping_projects.
