# ClaimRev

This repository contains the code associated with the following paper:


[Learning From Revisions: Quality Assessment of Claims in Argumentation at Scale](https://arxiv.org/pdf/2101.10250.pdf)
by Gabriella Skitalinskaya, Jonas Klaff, Henning Wachsmuth


## Reproducing results

### Models
You can download our finetuned models directly from [here](https://drive.google.com/drive/folders/1HdT9OX-gMOW0UQPCK27lHDAQgoTv_dDN?usp=sharing). Or you can finetune them as described below:

#### Classification

To train the bert model in a random split setup using the ClaimRev-BASE corpus:
```
python run_exp1_bert.py \
  --input_data './data/base.csv' \
  --pretrained_model 'bert-base-cased' \
  --batch_size 16 \
  --lr '1e-5' \
  --save_best 'True' \
  --output_dir './output/exp1_bert_random_base/' \
  --exp_setup 'random'
```
To train the sbert model in a random split setup using the ClaimRev-BASE corpus:
```
python run_exp1_sbert.py \
  --input_data './data/base.csv' \
  --pretrained_model 'bert-base-cased' \
  --batch_size 16 \
  --lr '1e-5' \
  --output_dir './output/exp1_sbert_random_base/' \
  --exp_setup 'random'
```
To change setup to cross-category, set the exp_setup argument to ```'cc'``` instead of ```'random'```. 
To use ClaimRev-EXT change the input_data argument to ```'./data/extended.csv' ```

#### Ranking

BTL ranking model (with *bert* and *sbert* accordingly):

```    
python run_btl.py \
    --prediction_dir './output/exp1_bert_random_base/' \
    --input_data './data/base.csv' \
    --output_file 'btl_bert.txt' \
    --exp_setup 'random' \
    --model 'bert'
```

```
python run_btl.py \
    --prediction_dir './output/exp1_sbert_random_base' \
    --input_data './data/base.csv' \
    --output_file 'btl_sbert.txt' \
    --exp_setup 'random' \
    --model 'sbert'
```
        
SVMRANK ranking model (with *bert* and *sbert* embeddings accordingly):

```
python run_svmrank.py \
    --input_data './data/full_list.csv'\
    --pretrained_model './output/exp1_bert_random_base' \
    --emb_output_file './data/emb/bert.csv' \
    --output_file 'svm_bert_base.txt' \
    --model_type 'bert'
```
 
```
python run_svmrank.py \
    --input_data './data/list_full.csv'\
    --pretrained_model './output/exp1_sbert_random_base' \
    --emb_output_file './data/emb/sbert.csv' \
    --output_file 'svm_sbert_base.txt'
    --model_type 'sbert'
```

Before running SVMRank-related experiments, follow the instructions to installing PySVMRank as described [here](https://github.com/ds4dm/PySVMRank).

### Data
In order to obtain access to the ClaimRev corpus, please reach out to Gabriella Skitalinskaya (email can be found in paper) along with your affiliation and a short description of how you will be using the data. Please let us know if you have any questions.

### Citation
If you use this corpus or code in your research, please include the following citation:
```
@inproceedings{skitalinskaya-etal-2021-learning,
    title = "Learning From Revisions: Quality Assessment of Claims in Argumentation at Scale",
    author = "Skitalinskaya, Gabriella  and
      Klaff, Jonas  and
      Wachsmuth, Henning",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.eacl-main.147",
    pages = "1718--1729",
}
```