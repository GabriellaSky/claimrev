import argparse
import pandas as pd
import re
from numpy import *
from scipy.stats import *
import os
import glob
import datetime
from sklearn.metrics import ndcg_score
from scipy.special import softmax
from sklearn.model_selection import train_test_split
import argparse

import warnings
warnings.filterwarnings(action='once')


class pairwise:
    def __init__(self):
        self.ctr = 0  # counts how many comparisons have been queried from the model

    def order(self):
        scores = self.scores()
        return argsort(-scores) + 1

    def scoring(self):
        scores = self.scores()
        return scores

    def sortP(self):
        scores = self.scores()

    def set_matrix(self, w):  # generates a Bradley-Terry-Luce model
        self.P = w
        self.sortP()

    def compare(self, i, j):  # draw a comparision from the model
        if i == j:
            print("does not make sense")
        self.ctr += 1
        if random.rand() < self.P[i, j]:
            return 1  # i beats j
        else:
            return 0  # j beats i

    def scores(self):
        P = array(self.P)
        for i in range(len(P)):
            P[i, i] = 0
        return sum(P, axis=1) / (len(self.P) - 1)

    def topk_complexity(self, k=1):
        sc = self.scores()
        lower = sum([1 / (sc[k - 1] - sc[i]) ** 2 for i in range(k, len(self.P))])
        upper = sum([1 / (sc[i] - sc[k]) ** 2 for i in range(0, k)])
        return lower + upper


if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", help="path to original data corpus", type=str, required=True)
    parser.add_argument("--prediction_dir", help="path to predictions", required=True, type=str)
    parser.add_argument("--output_file", help="path to store scores", required=True, type=str)
    parser.add_argument("--exp_setup", help="random split or cross category", required=True, type=str, default='random')
    parser.add_argument("--model", help="outputs from bert or sbert", required=True, type=str, default='bert')

    args = parser.parse_args()

    PRED_DIR = args.prediction_dir
    OUTPUT_FILE = args.output_file
    EXP_SETUP = args.exp_setup
    DATA = pd.read_csv(args.input_data)
    MODEL = args.model
    
    if EXP_SETUP == 'random':
        if MODEL == 'sbert':
        
            CLAIM_IDS = DATA.claim_id.unique()
            id_to_idx = DATA.groupby('claim_id')

            os.chdir(f'{PRED_DIR}')
            files = glob.glob('**logit.csv')
            res_list = []
            for temp_file in files:
                print(datetime.datetime.now())
                seed = int(temp_file.split('_')[-1].split('logit')[0])
                pred_data = pd.read_csv(temp_file, header=None, sep='\t')
                pred_data['logit'] = pred_data.apply(lambda x: [x[0], x[1]], axis=1)
                pred_data['softmax'] = pred_data.logit.apply(lambda x: softmax(x, axis=0))

                train_claim_ids, test_claim_ids = train_test_split(CLAIM_IDS, train_size=0.8, random_state=seed)

                test = DATA[(DATA.claim_id.isin(test_claim_ids))].reset_index() 
                test['softmax'] = pred_data.softmax

                # reorder so older is always first
                for index, row in test.iterrows():
                    if row.label == 0:
                        v1_text = row.v2_text
                        v2_text = row.v1_text
                        test.at[index, 'v1_text'] = v1_text
                        test.at[index, 'v2_text'] = v2_text
                        v1_id = row.v2_id
                        v2_id = row.v1_id
                        test.at[index, 'v1_id'] = v1_id
                        test.at[index, 'v2_id'] = v2_id
#                         test.at[index, 'pred_dense'] = 1
                        test.at[index, 'softmax'] = row.softmax[::-1]
                        test.at[index, 'label'] = 1
                test['v1_id'] = test.v1_id.apply(lambda x: int(x.split('.')[2]))
                test['v2_id'] = test.v2_id.apply(lambda x: int(x.split('.')[2]))
                pearson = []
                spearman = []
                kendal = []
                kendal_2 = []
                top_1 = []
                ndcg = []
                mrr = []
                place_1 = []
                for cur_id in test.claim_id.unique():
                    to_index = list(range(1, max(test[test.claim_id == cur_id]['v1_id'].unique()) + 2))
                    temp_data = test[test.claim_id == cur_id]
                    matrix = pd.DataFrame(index=to_index, columns=to_index)
                    for i in to_index:
                        for j in to_index:
                            if i == j:
                                matrix.at[i, j] = 0
                            else:
                                if i < j:
                                    query = temp_data[((temp_data.v1_id == i) & (temp_data.v2_id == j))].reset_index(drop=True)
                                    matrix.at[i, j] = query.softmax[0][0]
                                    matrix.at[j, i] = query.softmax[0][1]
                    blt_model = pairwise()
                    blt_model.set_matrix(matrix.to_numpy())
                    order = blt_model.order()
                    scores = list(reversed(blt_model.scoring()))
                    pearson.append(pearsonr(scores, list(reversed(range(1, len(order) + 1))))[0])
                    spearman.append(spearmanr(scores, list(reversed(range(1, len(order) + 1))))[0])
                    kendal.append(kendalltau(scores, list(reversed(range(1, len(order) + 1))))[0])
                    top_1.append(1 if order[0] == len(order) else 0)
                    place_1.append(list(order).index(len(order)) + 1)
                    ndcg.append(ndcg_score([list(reversed(range(1, len(order) + 1)))], [scores]))
                    mrr.append(1.0 / (list(order).index(len(order)) + 1))

                res = pd.DataFrame()
                res['id'] = test.claim_id.unique()
                res['pearson'] = pearson
                res['spearman'] = spearman
                res['kendal'] = kendal
                res['top_1'] = top_1
                res['ndcg'] = ndcg
                res['mrr'] = mrr
                res['place_1'] = place_1
                chain_len = pd.DataFrame(test.groupby(['claim_id'])['v1_id'].nunique() + 1).reset_index()
                res = pd.merge(chain_len, res, left_on='claim_id', right_on='id')
                res['bins'] = pd.cut(res['v1_id'], [0, 2, 3, 4, 5, 6, 30], labels=[1, 2, 3, 4, 5, '6+'])
                res['split'] = os.path.basename(temp_file).split('.')[0]
                res_list.append(res)

                full_res = pd.concat(res_list).reset_index(drop=True)
                
                # get mean scores for each group
                full_res.groupby(['split']).describe().unstack(1).loc[:, ("mean")].to_csv('btl_output.txt')

        else:
            
            os.chdir(f'{PRED_DIR}')
            files = glob.glob('**.csv')
            res_list = []
            for temp_file in files:
                print(datetime.datetime.now())
                pred_data = pd.read_csv(temp_file)
                pred_data['softmax'] = pred_data.pred.apply(lambda x: [float(a) for a in re.findall(r"[+-]?\d+(?:\.\d+)?", x)])
                test = pred_data
        
                # reorder so older is always first
                for index, row in test.iterrows():
                    if row.label == 0:
                        v1_text = row.v2_text
                        v2_text = row.v1_text
                        test.at[index, 'v1_text'] = v1_text
                        test.at[index, 'v2_text'] = v2_text
                        v1_id = row.v2_id
                        v2_id = row.v1_id
                        test.at[index, 'v1_id'] = v1_id
                        test.at[index, 'v2_id'] = v2_id
                        
#                         test.at[index, 'pred_dense'] = 1
                        test.at[index, 'softmax'] = row.softmax[::-1]
                        test.at[index, 'label'] = 1
                test['v1_id'] = test.v1_id.apply(lambda x: int(x.split('.')[2]))
                test['v2_id'] = test.v2_id.apply(lambda x: int(x.split('.')[2]))
                pearson = []
                spearman = []
                kendal = []
                kendal_2 = []
                top_1 = []
                ndcg = []
                mrr = []
                place_1 = []
                
                for cur_id in test.claim_id.unique():
                    to_index = list(range(1, max(test[test.claim_id == cur_id]['v1_id'].unique()) + 2))
                    temp_data = test[test.claim_id == cur_id]
                    matrix = pd.DataFrame(index=to_index, columns=to_index)
                    for i in to_index:
                        for j in to_index:
                            if i == j:
                                matrix.at[i, j] = 0
                            else:
                                if i < j:
                                    query = temp_data[((temp_data.v1_id == i) & (temp_data.v2_id == j))].reset_index(drop=True)
                                    print(query)
                                    matrix.at[i, j] = query.softmax[0][0]
                                    matrix.at[j, i] = query.softmax[0][1]
                    blt_model = pairwise()
                    blt_model.set_matrix(matrix.to_numpy())
                    order = blt_model.order()
                    scores = list(reversed(blt_model.scoring()))
                    pearson.append(pearsonr(scores, list(reversed(range(1, len(order) + 1))))[0])
                    spearman.append(spearmanr(scores, list(reversed(range(1, len(order) + 1))))[0])
                    kendal.append(kendalltau(scores, list(reversed(range(1, len(order) + 1))))[0])
                    top_1.append(1 if order[0] == len(order) else 0)
                    place_1.append(list(order).index(len(order)) + 1)
                    ndcg.append(ndcg_score([list(reversed(range(1, len(order) + 1)))], [scores]))
                    mrr.append(1.0 / (list(order).index(len(order)) + 1))
                    
                res = pd.DataFrame()
                res['id'] = test.claim_id.unique()
                res['pearson'] = pearson
                res['spearman'] = spearman
                res['kendal'] = kendal
                res['top_1'] = top_1
                res['ndcg'] = ndcg
                res['mrr'] = mrr
                res['place_1'] = place_1
                chain_len = pd.DataFrame(test.groupby(['claim_id'])['v1_id'].nunique() + 1).reset_index()
                res = pd.merge(chain_len, res, left_on='claim_id', right_on='id')
                res['bins'] = pd.cut(res['v1_id'], [0, 2, 3, 4, 5, 6, 30], labels=[1, 2, 3, 4, 5, '6+'])
                res['split'] = os.path.basename(temp_file).split('.')[0]
                res_list.append(res)

                full_res = pd.concat(res_list).reset_index(drop=True)
                
                # get mean scores for each group
                full_res.groupby(['split']).describe().unstack(1).loc[:, ("mean")].to_csv('btl_output.txt')

    else:
        if MODEL == 'sbert':
            CLAIM_IDS = DATA.claim_id.unique()
            id_to_idx = DATA.groupby('claim_id')

            os.chdir(f'{PRED_DIR}')
            files = glob.glob('**logit.csv')
            res_list = []
            for temp_file in files:
                print(datetime.datetime.now())
                category = temp_file.split('_')[-1].split('logit')[0][1]
                pred_data = pd.read_csv(temp_file, header=None, sep='\t')
                pred_data['logit'] = pred_data.apply(lambda x: [x[0], x[1]], axis=1)
                pred_data['softmax'] = pred_data.logit.apply(lambda x: softmax(x, axis=0))

                test = DATA[DATA[category] == 1].reset_index()  
                test['softmax'] = pred_data.softmax

                # reorder so older is always first
                for index, row in test.iterrows():
                    if row.label == 0:
                        v1_text = row.v2_text
                        v2_text = row.v1_text
                        test.at[index, 'v1_text'] = v1_text
                        test.at[index, 'v2_text'] = v2_text
                        v1_id = row.v2_id
                        v2_id = row.v1_id
                        test.at[index, 'v1_id'] = v1_id
                        test.at[index, 'v2_id'] = v2_id
#                         test.at[index, 'pred_dense'] = 1
                        test.at[index, 'softmax'] = row.softmax[::-1]
                        test.at[index, 'label'] = 1
                test['v1_id'] = test.v1_id.apply(lambda x: int(x.split('.')[2]))
                test['v2_id'] = test.v2_id.apply(lambda x: int(x.split('.')[2]))
                pearson = []
                spearman = []
                kendal = []
                kendal_2 = []
                top_1 = []
                ndcg = []
                mrr = []
                place_1 = []
                for cur_id in test.claim_id.unique():
                    to_index = list(range(1, max(test[test.claim_id == cur_id]['v1_id'].unique()) + 2))
                    temp_data = test[test.claim_id == cur_id]
                    matrix = pd.DataFrame(index=to_index, columns=to_index)
                    for i in to_index:
                        for j in to_index:
                            if i == j:
                                matrix.at[i, j] = 0
                            else:
                                if i < j:
                                    query = temp_data[((temp_data.v1_id == i) & (temp_data.v2_id == j))].reset_index(drop=True)
                                    matrix.at[i, j] = query.softmax[0][0]
                                    matrix.at[j, i] = query.softmax[0][1]
                    blt_model = pairwise()
                    blt_model.set_matrix(matrix.to_numpy())
                    order = blt_model.order()
                    scores = list(reversed(blt_model.scoring()))
                    pearson.append(pearsonr(scores, list(reversed(range(1, len(order) + 1))))[0])
                    spearman.append(spearmanr(scores, list(reversed(range(1, len(order) + 1))))[0])
                    kendal.append(kendalltau(scores, list(reversed(range(1, len(order) + 1))))[0])
                    top_1.append(1 if order[0] == len(order) else 0)
                    place_1.append(list(order).index(len(order)) + 1)
                    ndcg.append(ndcg_score([list(reversed(range(1, len(order) + 1)))], [scores]))
                    mrr.append(1.0 / (list(order).index(len(order)) + 1))

                res = pd.DataFrame()
                res['id'] = test.claim_id.unique()
                res['pearson'] = pearson
                res['spearman'] = spearman
                res['kendal'] = kendal
                res['top_1'] = top_1
                res['ndcg'] = ndcg
                res['mrr'] = mrr
                res['place_1'] = place_1
                chain_len = pd.DataFrame(test.groupby(['claim_id'])['v1_id'].nunique() + 1).reset_index()
                res = pd.merge(chain_len, res, left_on='claim_id', right_on='id')
                res['bins'] = pd.cut(res['v1_id'], [0, 2, 3, 4, 5, 6, 30], labels=[1, 2, 3, 4, 5, '6+'])
                res['split'] = os.path.basename(temp_file).split('.')[0]
                res_list.append(res)

                full_res = pd.concat(res_list).reset_index(drop=True)
                
                # get mean scores for each group
                full_res.groupby(['split']).describe().unstack(1).loc[:, ("mean")].to_csv('btl_sbert_cc.txt')
        else:
            
            os.chdir(f'{PRED_DIR}')
            files = glob.glob('**.csv')
            res_list = []
            for temp_file in files:
                print(datetime.datetime.now())
                pred_data = pd.read_csv(temp_file)
                pred_data['softmax'] = pred_data.pred.apply(lambda x: [float(a) for a in re.findall(r"[+-]?\d+(?:\.\d+)?", x)])
                test = pred_data
        
                # reorder so older is always first
                for index, row in test.iterrows():
                    if row.label == 0:
                        v1_text = row.v2_text
                        v2_text = row.v1_text
                        test.at[index, 'v1_text'] = v1_text
                        test.at[index, 'v2_text'] = v2_text
                        v1_id = row.v2_id
                        v2_id = row.v1_id
                        test.at[index, 'v1_id'] = v1_id
                        test.at[index, 'v2_id'] = v2_id
                        
#                         test.at[index, 'pred_dense'] = 1
                        test.at[index, 'softmax'] = row.softmax[::-1]
                        test.at[index, 'label'] = 1
                test['v1_id'] = test.v1_id.apply(lambda x: int(x.split('.')[2]))
                test['v2_id'] = test.v2_id.apply(lambda x: int(x.split('.')[2]))
                pearson = []
                spearman = []
                kendal = []
                kendal_2 = []
                top_1 = []
                ndcg = []
                mrr = []
                place_1 = []
                
                for cur_id in test.claim_id.unique():
                    to_index = list(range(1, max(test[test.claim_id == cur_id]['v1_id'].unique()) + 2))
                    temp_data = test[test.claim_id == cur_id]
                    matrix = pd.DataFrame(index=to_index, columns=to_index)
                    for i in to_index:
                        for j in to_index:
                            if i == j:
                                matrix.at[i, j] = 0
                            else:
                                if i < j:
                                    query = temp_data[((temp_data.v1_id == i) & (temp_data.v2_id == j))].reset_index(drop=True)
                                    print(query)
                                    matrix.at[i, j] = query.softmax[0][0]
                                    matrix.at[j, i] = query.softmax[0][1]
                    blt_model = pairwise()
                    blt_model.set_matrix(matrix.to_numpy())
                    order = blt_model.order()
                    scores = list(reversed(blt_model.scoring()))
                    pearson.append(pearsonr(scores, list(reversed(range(1, len(order) + 1))))[0])
                    spearman.append(spearmanr(scores, list(reversed(range(1, len(order) + 1))))[0])
                    kendal.append(kendalltau(scores, list(reversed(range(1, len(order) + 1))))[0])
                    top_1.append(1 if order[0] == len(order) else 0)
                    place_1.append(list(order).index(len(order)) + 1)
                    ndcg.append(ndcg_score([list(reversed(range(1, len(order) + 1)))], [scores]))
                    mrr.append(1.0 / (list(order).index(len(order)) + 1))
                    
                res = pd.DataFrame()
                res['id'] = test.claim_id.unique()
                res['pearson'] = pearson
                res['spearman'] = spearman
                res['kendal'] = kendal
                res['top_1'] = top_1
                res['ndcg'] = ndcg
                res['mrr'] = mrr
                res['place_1'] = place_1
                chain_len = pd.DataFrame(test.groupby(['claim_id'])['v1_id'].nunique() + 1).reset_index()
                res = pd.merge(chain_len, res, left_on='claim_id', right_on='id')
                res['bins'] = pd.cut(res['v1_id'], [0, 2, 3, 4, 5, 6, 30], labels=[1, 2, 3, 4, 5, '6+'])
                res['split'] = os.path.basename(temp_file).split('.')[0]
                res_list.append(res)

                full_res = pd.concat(res_list).reset_index(drop=True)
                
                # get mean scores for each group
                full_res.groupby(['split']).describe().unstack(1).loc[:, ("mean")].to_csv('btl_bert_cc.txt')
