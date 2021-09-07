import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers import models, losses
from sentence_transformers import InputExample
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from LabelAccuracyEvaluatorLogit import LabelAccuracyEvaluator
import math

if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", help="path to csv containing input data (comma separated)", required=True,
                        type=str)
    parser.add_argument("--n_epochs", help="Number of epochs", type=int, default=1)
    parser.add_argument('--batch_size', help='size of batch', type=int, default=32)
    parser.add_argument("--pretrained_model", help="Model to use to get unlabelled sample weights", type=str,
                        default='bert-base-cased')
    parser.add_argument("--seed", type=str, help="Random seed", default='401')
    parser.add_argument("--lr", type=str, help="set of learning rates", default='1e-5')
    parser.add_argument("--output_dir", help="where to save finetuned model and predictions", required=True, type=str)
    parser.add_argument("--exp_setup", type=str, help="random split or cross-category", default='random')

    args = parser.parse_args()

    LR = float(args.lr)
    SEED = int(args.seed)
    BATCH_SIZE = args.batch_size
    EPOCHS = args.n_epochs
    INPUT_DIR = args.input_data
    OUTPUT_DIR = args.output_dir
    MODEL = args.pretrained_model
    EXP_SETUP = args.exp_setup

    if EXP_SETUP == 'random':

        data = pd.read_csv(INPUT_DIR)
        CLAIM_IDS = data.claim_id.unique()
        id_to_idx = data.groupby('claim_id')

        word_embedding_model = models.Transformer(MODEL)

        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        train_claim_ids, test_claim_ids = train_test_split(CLAIM_IDS, train_size=0.8, random_state=SEED)

        train_data = data[(data.claim_id.isin(train_claim_ids))]
        train_samples = list(train_data.apply(lambda x: InputExample(texts=[x.v1_text, x.v2_text],
                                                                     label=x.label), axis=1))
        train_input = SentencesDataset(examples=train_samples, model=model)
        train_dataloader = DataLoader(train_input, shuffle=True, batch_size=BATCH_SIZE)
        train_loss = losses.SoftmaxLoss(model=model,
                                        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                        num_labels=2)

        test_data = data[data.claim_id.isin(test_claim_ids)]
        test_samples = list(test_data.apply(lambda x: InputExample(texts=[x.v1_text, x.v2_text],
                                                                   label=int(x.label)), axis=1))
        test_input = SentencesDataset(examples=test_samples, model=model)
        dev_dataloader = DataLoader(test_input, shuffle=False, batch_size=BATCH_SIZE)

        acc_evaluator = LabelAccuracyEvaluator(dev_dataloader, name=MODEL + '_' + str(SEED), softmax_model=train_loss)

        warmup_steps = math.ceil(len(train_dataloader) * EPOCHS * 0.1)  # 10% of train data for warm-up

        # Train the model
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=acc_evaluator,
                  epochs=EPOCHS,
                  warmup_steps=warmup_steps,
                  output_path=OUTPUT_DIR
                  )
    else:
        data = pd.read_csv(INPUT_DIR)
        CLAIM_IDS = data.claim_id.unique()
        id_to_idx = data.groupby('claim_id')

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

        for col in COLUMNS_TO_ITERATE:
            word_embedding_model = models.Transformer(MODEL)

            # Apply mean pooling to get one fixed sized sentence vector
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                           pooling_mode_mean_tokens=True,
                                           pooling_mode_cls_token=False,
                                           pooling_mode_max_tokens=False)

            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

            train = data[data[col] == 0]
            test_data = data[data[col] == 1]

            train_data = train
            train_samples = list(train_data.apply(lambda x: InputExample(texts=[x.v1_text, x.v2_text],
                                                                         label=x.label), axis=1))
            train_input = SentencesDataset(examples=train_samples, model=model)
            train_dataloader = DataLoader(train_input, shuffle=True, batch_size=BATCH_SIZE)
            train_loss = losses.SoftmaxLoss(model=model,
                                            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                            num_labels=2)

            test_samples = list(test_data.apply(lambda x: InputExample(texts=[x.v1_text, x.v2_text],
                                                                       label=int(x.label)), axis=1))
            test_input = SentencesDataset(examples=test_samples, model=model)
            dev_dataloader = DataLoader(test_input, shuffle=False, batch_size=BATCH_SIZE)

            acc_evaluator = LabelAccuracyEvaluator(dev_dataloader,
                                                   name=MODEL + "_" + str(col),
                                                   softmax_model=train_loss)

            warmup_steps = math.ceil(len(train_dataloader) * EPOCHS * 0.1)  # 10% of train data for warm-up

            # Train the model
            model.fit(train_objectives=[(train_dataloader, train_loss)],
                      evaluator=acc_evaluator,
                      epochs=EPOCHS,
                      warmup_steps=warmup_steps,
                      output_path=OUTPUT_DIR
                      )