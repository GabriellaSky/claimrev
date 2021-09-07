import argparse
import os
import svmrank
from sklearn import preprocessing
from scipy.stats import *
from sklearn.metrics import ndcg_score
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
from transformers import *
from tqdm import *

import warnings

warnings.filterwarnings('ignore')


def get_sbert_embeddings(model_path, data_path, output_emb_path):
	data = pd.read_csv(data_path)
	model = SentenceTransformer(model_path)
	sentence_embeddings = model.encode(data.claim_text)
	emb = pd.DataFrame(sentence_embeddings).apply(pd.Series)
	emb = emb.add_prefix('feature_')
	temp = pd.concat([data, emb], axis=1)

	if not os.path.exists(output_emb_path):
		os.makedirs(output_emb_path)
	temp.to_csv(f'{output_emb_path}')


def get_bert_embeddings(model_path, data_path, output_emb_path):
	tokenizer = AutoTokenizer.from_pretrained(model_path)
	model = AutoModel.from_pretrained(model_path, from_tf=True)

	# read dataset
	dataset = pd.read_csv(data_path)
	# init embedding
	embedding = TransformerDocumentEmbeddings(model, fine_tune=False)
	embs = []
	for i in tqdm_notebook(list(dataset.claim_text)):
		sentence = Sentence(i)
		# embed the sentence
		embedding.embed(sentence)
		embs.append(sentence.embedding.cpu().numpy())

	features = pd.DataFrame(embs).apply(pd.Series)
	features = features.add_prefix('feature_')
	features.to_csv(f'{output_emb_path}')


def run_svm_rank(data, state):
	CLAIM_IDS = data.claim_id.unique()

	train_claim_ids, test_claim_ids = train_test_split(CLAIM_IDS, train_size=0.8, random_state=state)

	train = data[data.claim_id.isin(train_claim_ids)]
	test = data[data.claim_id.isin(test_claim_ids)]

	filter_train = [col for col in train if col.startswith('feature')]
	filter_test = [col for col in test if col.startswith('feature')]

	train_xs = np.array(train[filter_train])
	train_ys = np.array(train.revision_id)

	le = preprocessing.LabelEncoder()
	train_groups = le.fit_transform(train.claim_id)

	test_xs = np.array(test[filter_test])
	test_groups = le.fit_transform(test.claim_id)

	m = svmrank.Model({'-c': 3})
	m.fit(train_xs, train_ys, train_groups)

	test['pred'] = m.predict(test_xs, test_groups)
	names = 'sbert_' + str(state)
	test[['claim_id', 'revision_id', 'pred']].to_csv(names + ".csv")

	return run_eval(names, test)


def run_eval(files, test):
	print("--Evaluating--")

	pearson = []
	spearman = []
	kendal = []
	top_1 = []
	ndcg = []
	mrr = []
	for cur_id in test.claim_id.unique():
		temp_data = test[test.claim_id == cur_id].sort_values('revision_id')
		pearson.append(pearsonr(temp_data.revision_id, temp_data.pred)[0])
		spearman.append(spearmanr(temp_data.revision_id, temp_data.pred)[0])
		kendal.append(kendalltau(temp_data.revision_id, temp_data.pred)[0])
		top_1.append(1 if (list(temp_data.pred).index(max(temp_data.pred)) + 1) == len(temp_data.pred) else 0)
		ndcg.append(ndcg_score([temp_data.revision_id], [temp_data.pred]))
		mrr.append(1.0 / (list(reversed(list(temp_data.pred))).index(max(temp_data.pred)) + 1))

	res = pd.DataFrame()
	res['sess_id'] = test.claim_id.unique()
	res['pearson'] = pearson
	res['spearman'] = spearman
	res['kendal'] = kendal
	res['top_1'] = top_1
	res['ndcg'] = ndcg
	res['mrr'] = mrr
	chain_len = pd.DataFrame(test.groupby(['claim_id'])['revision_id'].nunique()).reset_index()
	res = pd.merge(chain_len, res, left_on='claim_id', right_on='sess_id')
	res['bins'] = pd.cut(res['revision_id'], [0, 2, 3, 4, 5, 6, 30], labels=[1, 2, 3, 4, 5, '6+'])
	res['topic'] = files

	return res


if __name__ == "__main__":
	# Define arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_data", help="path to csv containing input data (comma separated)", required=True,
						type=str)
	parser.add_argument("--pretrained_model", help="Model to use to get unlabelled sample weights", type=str,
						default='bert-base-cased')
	parser.add_argument("--emb_output_file", help="where to save embeddings for future reuse", required=True, type=str)
	parser.add_argument("--output_file", help="where to save scores", required=True, type=str)
	parser.add_argument("--seed", type=str, help="Random seed", default='401')
	parser.add_argument("--exp_setup", type=str, help="random split or cross-category", default='random')
	parser.add_argument("--model_type", type=str, help="bert or sbert", default='sbert')

	args = parser.parse_args()

	INPUT_DATA = args.input_data
	EMB_OUTPUT_DIR = args.emb_output_file
	MODEL = args.pretrained_model
	SEED = int(args.seed)
	OUTPUT_FILE = args.output_file
	EXP_SETUP = args.exp_setup
	MODEL_TYPE = args.model_type
	COLUMNS_TO_ITERATE = [
		'Children',
		'ClimateChange',
		'Democracy',
		'Economics',
		'Education',
		'Equality',
		'Ethics',
		'Europe',
		'Gender',
		'Government',
		'Health',
		'Justice',
		'Law',
		'Philosophy',
		'Politics',
		'Religion',
		'Science',
		'Society',
		'Technology',
		'USA'
	]

	# generate proper embeddings
	if MODEL_TYPE == 'sbert':
		get_sbert_embeddings(MODEL, INPUT_DATA, EMB_OUTPUT_DIR)
	else:
		get_bert_embeddings(MODEL, INPUT_DATA, EMB_OUTPUT_DIR)

	dataset = pd.read_csv(INPUT_DATA)
	num_queries = len(dataset.claim_id.unique())
	features = pd.read_csv(f'{EMB_OUTPUT_DIR}')
	features = features.drop(['claim_id', 'revision_id'], axis=1)

	temp = pd.concat([dataset.reset_index(drop=True), features.reset_index(drop=True)], axis=1)

	if EXP_SETUP == 'cc':
		acc = []
		mccs = []
		results = []

		res_list = []
		for col in COLUMNS_TO_ITERATE:
			print("Starting: " + col)

			train = temp[temp[col] == 0]
			test = temp[temp[col] == 1]

			filter_train = [col for col in train if col.startswith('feature')]
			filter_test = [col for col in test if col.startswith('feature')]

			train_xs = np.array(train[filter_train])
			train_ys = np.array(train.revision_id)

			le = preprocessing.LabelEncoder()
			train_groups = le.fit_transform(train.claim_id)

			test_xs = np.array(test[filter_test])
			test_ys = np.array(test.revision_id)
			test_groups = le.fit_transform(test.claim_id)

			m = svmrank.Model({'-c': 3})
			m.fit(train_xs, train_ys, train_groups)

			test['pred'] = m.predict(test_xs, test_groups)
			names = 'cc' + str(col)
			test[['changes_affectedIds_id', 'revision_id', 'pred']].to_csv(names + ".csv")

			res_list.append(run_eval(names, test))

		full_res = pd.concat(res_list).reset_index(drop=True)
		full_res.groupby(['split']).describe().unstack(1).loc[:, ("mean")].to_csv('svmrank_output_cc.txt')

		print('Pearson correlation: ', full_res.groupby(['split']).describe()['pearson']['mean'].mean())
		print('Spearman correlation: ', full_res.groupby(['split']).describe()['spearman']['mean'].mean())
		print('Top-1: ', full_res.groupby(['split']).describe()['top_1']['mean'].mean())
		print('MRR: ', np.mean(full_res.groupby(['topic'])['mrr'].sum() / num_queries))

	else:
		res_list = []
		res_list.append(run_svm_rank(temp, SEED))

		full_res = pd.concat(res_list).reset_index(drop=True)
		full_res.groupby(['split']).describe().unstack(1).loc[:, ("mean")].to_csv('svmrank_output_random.txt')

		print('Pearson correlation: ', full_res.groupby(['split']).describe()['pearson']['mean'].mean())
		print('Spearman correlation: ', full_res.groupby(['split']).describe()['spearman']['mean'].mean())
		print('Top-1: ', full_res.groupby(['split']).describe()['top_1']['mean'].mean())
		print('MRR: ', np.mean(full_res.groupby(['topic'])['mrr'].sum() / num_queries))
