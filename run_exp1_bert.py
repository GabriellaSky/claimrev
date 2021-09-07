import random
import argparse
import tensorflow as tf
from transformers import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef
# import torch
import os
import numpy as np
import logging as lg
lg.basicConfig()


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# logging.set_verbosity_info()


def set_seed(seed):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf``.

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_tf_available():
        tf.random.set_seed(seed)


def get_model(lr, model, epsilon, clipnorm):
    tokenizer = BertTokenizer.from_pretrained(model)
    model = TFBertForSequenceClassification.from_pretrained(model)

    # Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=epsilon, clipnorm=clipnorm)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model, tokenizer, optimizer, loss, metrics


def get_logger(logger_name, model_name, output_dir, create_file=False):
    # create logger for prd_ci
    log = lg.getLogger(logger_name)
    log.setLevel(level=lg.DEBUG)

    # create formatter and add it to the handlers
    formatter = lg.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler for logger.
    fh = lg.FileHandler(f'{output_dir}exp1_{model_name}_random.log')
    fh.setLevel(level=lg.DEBUG)
    fh.setFormatter(formatter)

    # create console handler for logger.
    ch = lg.StreamHandler()
    ch.setLevel(level=lg.DEBUG)
    ch.setFormatter(formatter)

    # add handlers to logger.
    if create_file:
        log.addHandler(fh)

    log.addHandler(ch)
    return log


def get_train_test_split(train, test_columns):
    test = pd.DataFrame()
    for col in test_columns:
        test = test.append(train.loc[train[col] == 1])
        train = train[train[col] != 1]

    return train, test


class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('The average loss for epoch {} is {:7.2f} and train-acc is {:7.2f}.'.format(epoch, logs['loss'],
                                                                                                 logs['accuracy']))
        print(
            'The average val-loss for epoch {} is {:7.2f} and val-acc is {:7.2f}.'.format(epoch, logs['val_loss'],
                                                                                          logs['val_accuracy']))


def generateTfDataset(features):
    def gen():
        for ex in features:
            yield (
                {
                    "input_ids": ex.input_ids,
                    "attention_mask": ex.attention_mask,
                    "token_type_ids": ex.token_type_ids,
                },
                ex.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", help="Positive samples with citations", required=True, type=str)
    parser.add_argument("--n_epochs", help="Number of epochs", type=int, default=2)
    parser.add_argument("--pretrained_model", help="Model to use to get unlabelled sample weights", type=str,
                        default='bert-base-cased')
    parser.add_argument("--seed", type=str, help="Random seed", default='401')
    parser.add_argument("--lr", type=str, help="set of learning rates", default='1e-5')
    parser.add_argument("--save_best", help="save best model",  type=str, default='True')
    parser.add_argument("--save_pred", help="save predictions", type=str, default='True')
    parser.add_argument("--output_dir", help="where to save finetuned model and predictions", required=True, type=str)
    parser.add_argument('--log', help='use logger to monitor', type=bool, default=True)
    parser.add_argument('--epsilon', help='epsilon', type=float, default=1e-08)
    parser.add_argument('--clipnorm', help='clipnorm', type=float, default=1.0)
    parser.add_argument('--batch_size', help='size of batch', type=int, default=16)
    parser.add_argument('--max_seq_len', help='maximum length of sequence', type=int, default=128)
    parser.add_argument("--exp_setup", type=str, help="random split or cross-category", default="random")
    parser.add_argument("--save_model", help="save model for each category(only for cross category setup",  type=bool, default=True)

    args = parser.parse_args()

    LR = float(args.lr)
    SEED = int(args.seed)
    SAVE_MODEL = bool(args.save_best)
    SAVE_PREDICTIONS = bool(args.save_pred)
    EPOCHS = args.n_epochs
    INPUT_DIR = args.input_data
    OUTPUT_DIR = args.output_dir
    MODEL = args.pretrained_model
    USE_LOGGER = args.log
    EPSILON = args.epsilon
    CLIPNORM = args.clipnorm
    MAX_SEQ_LEN = args.max_seq_len
    BATCH_SIZE = args.batch_size
    EXP_SETUP = args.exp_setup

    if EXP_SETUP == 'random':

        if USE_LOGGER:
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            logger = get_logger("Custom logger", MODEL, OUTPUT_DIR, create_file=True)
            logger.debug(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")
            strategy = tf.distribute.MirroredStrategy()
            logger.debug('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        df = pd.read_csv(f'{INPUT_DIR}')

        CLAIM_IDS = df.claim_id.unique()
        id_to_idx = df.groupby('claim_id')

        examples = [InputExample(guid=index,
                                 text_a=row['v1_text'],
                                 text_b=row['v2_text'],
                                 label=str(row['label'])) for index, row in df.iterrows()]

        _, tokenizer, _, _, _ = get_model(1e-5, MODEL, EPSILON, CLIPNORM)

        dataset = glue_convert_examples_to_features(examples, tokenizer, max_length=MAX_SEQ_LEN, task='mrpc')

        if USE_LOGGER:
            logger.debug(f"{SEED}")

        set_seed(SEED)

        model, tokenizer, optimizer, loss, metrics = get_model(LR, MODEL, EPSILON, CLIPNORM)
        train_claim_ids, test_claim_ids = train_test_split(CLAIM_IDS, train_size=0.8, random_state=SEED)

        train_ids = [index for claim_id in train_claim_ids for index in id_to_idx.groups[claim_id]]
        test_ids = [index for claim_id in test_claim_ids for index in id_to_idx.groups[claim_id]]

        train = [dataset[i] for i in train_ids]
        test = [dataset[i] for i in test_ids]

        print('Train: {}'.format(len(train)))
        print('Test: {}'.format(len(test)))
        print('Seed: {}'.format(SEED))
        print('LR: {}'.format(LR))

        if USE_LOGGER:
            logger.debug('Train: {}'.format(len(train)))
            logger.debug('Test: {}'.format(len(test)))
            logger.debug('Seed: {}'.format(SEED))
            logger.debug('LR: {}'.format(LR))

        train_dataset = generateTfDataset(train)
        test_dataset = generateTfDataset(test)

        # Prepare dataset for GLUE as a tf.data.Dataset instance
        train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE)
        valid_dataset = test_dataset.batch(BATCH_SIZE)

        for epoch in range(EPOCHS):

            # Train and evaluate using tf.keras.Model.fit()
            history = model.fit(train_dataset, epochs=1, validation_data=valid_dataset)

            print(f"Result for seed {SEED}:")
            print(f"accuracy: {history.history['accuracy']} loss: {history.history['loss']}")
            print(f"val_accuracy: {history.history['val_accuracy']} val_loss: {history.history['val_loss']}")

            if USE_LOGGER:
                logger.debug('Epoch: {}'.format(epoch))
                logger.debug(f"Result for seed {SEED}:")
                logger.debug(f"accuracy: {history.history['accuracy']} loss: {history.history['loss']}")
                logger.debug(
                    f"val_accuracy: {history.history['val_accuracy']} val_loss: {history.history['val_loss']}")

            test_labels = [entry.label for entry in test]
            y_pred = model.predict(valid_dataset)

            y_pred_dense = np.argmax(y_pred, axis=1)
            m = tf.keras.metrics.SparseCategoricalAccuracy()
            _ = m.update_state(test_labels, y_pred)

            print(f"SparseCategoricalAccuracy: {m.result().numpy()}")
            print(f"Macro P/R/F1: {precision_recall_fscore_support(test_labels, y_pred_dense, average='macro')}")
            print(f"Micro P/R/F1:  {precision_recall_fscore_support(test_labels, y_pred_dense, average='micro')}")
            print(
                f"Weighted P/R/F1:  {precision_recall_fscore_support(test_labels, y_pred_dense, average='weighted')}")
            print(f"MCC: {matthews_corrcoef(test_labels, y_pred_dense, sample_weight=None)}")

            if USE_LOGGER:
                logger.debug(f"SparseCategoricalAccuracy: {m.result().numpy()}")
                logger.debug(
                    f"Macro P/R/F1: {precision_recall_fscore_support(test_labels, y_pred_dense, average='macro')}")
                logger.debug(
                    f"Micro P/R/F1:  {precision_recall_fscore_support(test_labels, y_pred_dense, average='micro')}")
                logger.debug(
                    f"Weighted P/R/F1:  {precision_recall_fscore_support(test_labels, y_pred_dense, average='weighted')}")
                logger.debug(f"MCC: {matthews_corrcoef(test_labels, y_pred_dense, sample_weight=None)}")

            if epoch == EPOCHS - 1:
                if SAVE_PREDICTIONS:
                    output = df[df.index.isin(test_ids)].copy().reindex(test_ids)
                    output['pred_dense'] = y_pred_dense
                    output['pred'] = list(y_pred)

                    directory = f'{OUTPUT_DIR}/lr_{str(LR)}'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    output.to_csv(directory + "/seed_" + str(SEED) + "_predictions.csv")

        if SAVE_MODEL:
            print("Saving model...")
            if USE_LOGGER:
                logger.debug("Saving model...")
            # Save model
            directory = f'{OUTPUT_DIR}/lr_{str(LR)}'
            if not os.path.exists(directory):
                os.makedirs(directory)
            model.save_pretrained(directory + '/')
    else:
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

        if USE_LOGGER:
            logger = get_logger("Custom logger", MODEL, OUTPUT_DIR, create_file=True)
            logger.debug(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")
            strategy = tf.distribute.MirroredStrategy()
            logger.debug('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        df = pd.read_csv(f'{INPUT_DIR}')

        CLAIM_IDS = df.changes_affectedIds_id.unique()
        id_to_idx = df.groupby('claim_id')

        examples = [InputExample(guid=index,
                                 text_a=row['v1_text'],
                                 text_b=row['v2_text'],
                                 label=str(row['label'])) for index, row in df.iterrows()]

        _, tokenizer, _, _, _ = get_model(LR, MODEL, EPSILON, CLIPNORM)

        dataset = glue_convert_examples_to_features(examples, tokenizer, max_length=MAX_SEQ_LEN, task='mrpc')

        for column in COLUMNS_TO_ITERATE:
            try:

                if USE_LOGGER:
                    logger.debug(f"Start preparing data for {column}")

                train, test = get_train_test_split(df.copy(), [column])
                CURRENT_OUTPUT_DIRECTORY = f'{OUTPUT_DIR}'

                if USE_LOGGER:
                    logger.debug('Train: {}'.format(train.shape))
                    logger.debug('Test: {}'.format(test.shape))
                    logger.debug(f"Building and Compiling {MODEL} Model...")

                model, tokenizer, optimizer, loss, metrics = get_model(LR, MODEL, EPSILON, CLIPNORM)

                train_idx = df.index[df[column] != 1].to_list()
                test_idx = df.index[df[column] == 1].to_list()

                train_dataset = [dataset[i] for i in train_idx]
                valid_dataset = [dataset[i] for i in test_idx]

                train_dataset = generateTfDataset(train_dataset)
                test_dataset = generateTfDataset(valid_dataset)

                # Prepare dataset for GLUE as a tf.data.Dataset instance
                train_dataset = train_dataset.cache().shuffle(1000).batch(BATCH_SIZE)
                valid_dataset = test_dataset.batch(BATCH_SIZE)

                if USE_LOGGER:
                    logger.debug("Start training:")

                history = model.fit(train_dataset,
                                    epochs=EPOCHS,
                                    validation_data=valid_dataset,
                                    verbose=1,
                                    callbacks=[LossAndErrorPrintingCallback()])

                y_pred = model.predict(valid_dataset)
                y_pred_dense = np.argmax(y_pred, axis=1)
                m = tf.keras.metrics.SparseCategoricalAccuracy()
                _ = m.update_state(test.label.values, y_pred)

                print(f"SparseCategoricalAccuracy: {m.result().numpy()}")
                print(
                    f"Macro P/R/F1: {precision_recall_fscore_support(test.label.values, y_pred_dense, average='macro')}")
                print(
                    f"Micro P/R/F1:  {precision_recall_fscore_support(test.label.values, y_pred_dense, average='micro')}")
                print(
                    f"Weighted P/R/F1:  {precision_recall_fscore_support(test.label.values, y_pred_dense, average='weighted')}")
                print(f"MCC: {matthews_corrcoef(test.label.values, y_pred_dense, sample_weight=None)}")

                if USE_LOGGER:
                    logger.debug(f"SparseCategoricalAccuracy: {m.result().numpy()}")
                    logger.debug(
                        f"Macro P/R/F1: {precision_recall_fscore_support(test.label.values, y_pred_dense, average='macro')}")
                    logger.debug(
                        f"Micro P/R/F1:  {precision_recall_fscore_support(test.label.values, y_pred_dense, average='micro')}")
                    logger.debug(
                        f"Weighted P/R/F1:  {precision_recall_fscore_support(test.label.values, y_pred_dense, average='weighted')}")
                    logger.debug(f"MCC: {matthews_corrcoef(test.label.values, y_pred_dense, sample_weight=None)}")

                output = df[df.index.isin(test_idx)].copy().reindex(test_idx)
                output['pred_dense'] = y_pred_dense
                output['pred'] = list(y_pred)

                if SAVE_PREDICTIONS:
                    if not os.path.exists(CURRENT_OUTPUT_DIRECTORY):
                        os.makedirs(CURRENT_OUTPUT_DIRECTORY)
                    output.to_csv(CURRENT_OUTPUT_DIRECTORY + "/" + column + "_" + str(SEED) + "_pred.csv")

                if SAVE_MODEL:
                    if USE_LOGGER:
                        logger.debug(f"Saving model to {CURRENT_OUTPUT_DIRECTORY}/{column}")
                    model.save_pretrained(f"{CURRENT_OUTPUT_DIRECTORY}/{column}")

            except Exception as e:
                if USE_LOGGER:
                    logger.exception(f"Failed for column {column}")
                continue