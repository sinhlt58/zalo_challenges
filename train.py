import time
import os
import sys
from ast import literal_eval
import csv

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd

from models import get_model
from utils import get_bert_paths, create_folder, convert_df_str, read_json_data


class Trainer(object):

    def __init__(
        self,
        lang,
        data_folder,
        model_type,
        lr,
        current_epoch=0,
        input_feature_cols=["input_features"]
    ):
        self.lang = lang
        self.data_folder = data_folder
        self.model_type = model_type
        self.current_epoch = current_epoch
        self.input_feature_cols = input_feature_cols

        # bert model path for en
        if lang == "en":
            bert_model_path, _, _ = get_bert_paths(
                data_folder.split("_")[-1]
            )
        else:
            bert_model_path = None

        # combine from data_folder and model_type
        save_folder = "{}_{}".format(model_type, data_folder)
        self.checkpoint_path = "models/{}".format(
            save_folder
        )
        create_folder(self.checkpoint_path)

        saved_epoch_path = None
        if current_epoch > 0:
            saved_epoch = current_epoch - 1
            saved_epoch_path = "{}/{}/{}".format(
                self.checkpoint_path, saved_epoch, saved_epoch
            )

            # create report folder
            self.report_folder = "reports/{}/{}".format(
                save_folder, saved_epoch
            )
            create_folder(self.report_folder)

        # get configs
        configs_file = "qna_data/pre_data/{}/configs.json".format(data_folder)
        configs = read_json_data(configs_file)

        self.model = get_model(
            lang=lang,
            model_type=model_type,
            bert_model_path=bert_model_path,
            saved_epoch_path=saved_epoch_path,
            configs=configs
        )

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr
        )

    def _print_sizes(self):
        print ("train_size: ", self.train_size)
        print ("dev_size: ", self.dev_size)
        print ("batch_size: ", self.batch_size)

    def sanity_check(self):
        self._load_data(
            data_folder=self.data_folder,
            dataset_type="train",
            num_sample=2000,
            batch_size=32,
            split_ratio=0.1,
        )

        self._print_sizes()
        example_batches = self.train_set.take(1)

        self.model.summary()

        for example_batch in example_batches:
            model_inputs = self._get_model_inputs(example_batch, self.input_feature_cols)
            print ("model_inputs: ", model_inputs)

            example_out = self.model(model_inputs)
            print ("example logits: ", example_out)

            labels = tf.cast(example_batch["label"], tf.float32)
            loss = self._train_step(
                self.model, self.optimizer, model_inputs, labels
            )
            print ("example loss: ", loss)

            example_pred, example_probs = self._predict(self.model, example_batch)
            print ("example labels: ", labels)
            print ("example_pred: ", example_pred)
            print ("example_probs: ", example_probs)

            accuracy, f1 = self._score(self.model, example_batch)
            print ("example accuracy {:.4f}".format(accuracy))
            print ("example f1 {:.4f}".format(f1))

            dev_accuracy, dev_f1 = self._score_dataset(self.model, self.dev_set, self.dev_size)
            print ("dev accuracy {:.4f}".format(dev_accuracy))
            print ("dev f1 {:.4f}".format(dev_f1))


    def train_dev(self, epochs):
        self._load_data(
            data_folder=self.data_folder,
            dataset_type="train",
            num_sample=2000,
            batch_size=128,
            split_ratio=0.1,
        )
        self._print_sizes()

        self._train(epochs)

    def _load_full_data(self):
        self._load_data(
            data_folder=self.data_folder,
            dataset_type="train",
            num_sample=None, # use full data
            batch_size=64,
            split_ratio=0.2,
        )
        self._print_sizes()

    def train_full(self, epochs):
        self._load_full_data()
        self._train(epochs, save_model=True)

    def eval(self):
        # self._load_full_data()

        # load the test dataset
        self._load_data(
            data_folder=self.data_folder,
            dataset_type="test",
        )

        # dev_metrics, dev_results = self._score_dataset(
        #     model=self.model,
        #     dataset=self.dev_set,
        #     size=self.dev_size,
        #     mode="eval",
        # )
        test_metrics, test_results = self._score_dataset(
            model=self.model,
            dataset=self.test_set,
            size=self.test_size,
            mode="eval",
        )

        # self._print_metrics_results("dev", dev_metrics, dev_results)
        self._print_metrics_results("test", test_metrics, test_results)

    def _get_model_inputs(self, batch_sample, feature_colums):
        model_inputs = []
        for col in feature_colums:
            model_inputs.append(batch_sample[col])
        return model_inputs

    def _print_metrics_results(self, dataset_type, metrics, results):
        accuracy, f1 = metrics

        print ("{} accuracy {:.4f}".format(
            dataset_type, accuracy
        ))
        print ("{} f1 {:.4f}".format(
            dataset_type, f1
        ))

        results_csv = "{}/{}_results.csv".format(
            self.report_folder, dataset_type
        )
        results_df = pd.DataFrame(data=results)
        convert_df_str(results_df)
        results_df.to_csv(results_csv, encoding="utf-8", index=False)
        print ("Done write results file {}".format(results_csv))

        if dataset_type in ["test", "private", "dev"]:
            submit_csv = "{}/{}_submit.csv".format(
                self.report_folder, dataset_type
            )

            submit_df = results_df.loc[results_df["pred"] == 1]
            submit_df = submit_df[["id", "pid"]]

            submit_df.to_csv(submit_csv, header=["test_id", "answer"], encoding="utf-8", index=False)

            print ("Done write submit file to {}".format(submit_csv))

    def _get_step_per_epoch(self, size, batch_size):
        res = size // batch_size
        if size % batch_size:
            res += 1
        return res

    def _train(self, epochs, save_model=False):
        steps_per_epoch = self._get_step_per_epoch(self.train_size, self.batch_size)
        # print ("_train steps_per_epoch: ", steps_per_epoch)

        for epoch in range(self.current_epoch, epochs):
            start = time.time()

            total_loss = 0

            for (batch, batch_sample) in enumerate(self.train_set.take(steps_per_epoch)):
                model_inputs = self._get_model_inputs(batch_sample, self.input_feature_cols)
                batch_labels = tf.cast(batch_sample["label"], tf.float32)

                batch_loss = self._train_step(
                    self.model, self.optimizer, model_inputs, batch_labels
                )

                # if batch % 10 == 0:
                #     print ("Epoch {} batch {}/{} with loss {}".format(
                #         epoch, batch, steps_per_epoch, batch_loss
                #     ))

                total_loss += batch_loss

            print ("Done epoch {}".format(epoch))
            dev_accuracy, dev_f1 = self._score_dataset(self.model, self.dev_set, self.dev_size)
            train_accuracy, train_f1 = self._score_dataset(self.model, self.train_set, self.dev_size)

            print (
                'Epoch {} Loss {:.4f}. Train accuracy {:.4f} f1 {:.4f}.'
                'Dev accuracy {:.4f} f1 {:.4f}. Time taken for 1 epoch {:.4f} sec\n'.format(
                    epoch, total_loss / steps_per_epoch, train_accuracy, train_f1,
                    dev_accuracy, dev_f1,
                    time.time() - start
                )
            )

            if epoch and epoch % 10 == 0 and save_model:
                checkpoint_path = "{}/{}".format(
                    self.checkpoint_path, epoch
                )
                create_folder(checkpoint_path)
                checkpoint_path = "{}/{}".format(
                    checkpoint_path, epoch
                )
                print ("Saved model to path: ", checkpoint_path)
                self.model.save_weights(checkpoint_path)

    def _score_dataset(
        self, model, dataset, size, threshold=0.5, mode="train",
    ):
        labels = []
        pred = []
        probs = []

        results = {
            "id": [],
            "pid": [],
        }

        steps_per_epoch = self._get_step_per_epoch(size, self.batch_size)
        # print ("_score_dataset steps_per_epoch: ", steps_per_epoch)

        for batch_sample in dataset.take(steps_per_epoch):
            batch_pred, batch_probs = self._predict(model, batch_sample, threshold)
            batch_labels = batch_sample["label"]

            batch_pred = batch_pred.astype(int).tolist()
            batch_labels = batch_labels.numpy().astype(int).tolist()

            # print ("batch_pred: ", batch_pred)
            # print ("batch_labels: ", batch_labels)
            if mode == "eval":
                results["id"] += batch_sample["id"].numpy().tolist()
                results["pid"] += batch_sample["pid"].numpy().tolist()

            labels += batch_labels
            pred += batch_pred
            probs += batch_probs.tolist()

        labels = np.array(labels)
        pred = np.array(pred)

        # print ("labels: ", labels)
        # print ("pred: ", pred)

        metrics = self._calculate_metrics(labels, pred)
        if mode == "train":
            return metrics

        elif mode == "eval":
            results["label"] = labels.tolist()
            results["pred"] = pred.tolist()
            results["prob"] = probs

            return metrics, results
        else:
            print ("_score_dataset: invalid mode {}".format(mode))
            raise Exception

    def _calculate_metrics(self, labels, pred):
        accuracy = (labels == pred).sum() / len(labels)
        f1 = f1_score(labels, pred)

        return accuracy, f1

    def _score(self, model, batch_sample, threshold=0.5):
        batch_pred, _ = self._predict(model, batch_sample, threshold)
        batch_labels = batch_sample["label"]

        batch_pred = batch_pred.astype(int)
        batch_labels = batch_labels.numpy().astype(int)

        return self._calculate_metrics(batch_labels, batch_pred)

    def _predict(self, model, batch_sample, threshold=0.5):
        model_inputs = self._get_model_inputs(batch_sample, self.input_feature_cols)

        batch_logits = model(model_inputs, training=False)
        batch_probs = tf.math.sigmoid(batch_logits)

        batch_pred = batch_probs > threshold
        batch_pred = tf.cast(batch_pred, tf.float32)

        return batch_pred.numpy(), batch_probs.numpy()

    def _loss_function(self, logits, labels):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits,
            labels=labels,
        )

        return tf.math.reduce_mean(loss)

    def _train_step(self, model, optimizer, model_inputs, labels):
        # NOTE: We only watch for trainable_variables in the model
        # we ignore all the untrainale variables like variables from bert
        # for optimization
        # print ("trainable variables: ", model.trainable_variables)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # We only watch trainable variables
            for var in model.trainable_variables:
                tape.watch(var)

            logits = model(model_inputs)
            loss = self._loss_function(logits, labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        # print ("gradients[0]: ", gradients[0])

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss

    def _summary_data(self, df, dataset_type):
        n = df.shape[0]

        num_true = (df["label"] == 1).sum()
        num_false = n - num_true

        print ("Summary {} dataset".format(dataset_type))
        print ("Num of true label {}/{} - {:.4f}".format(num_true, n, num_true/n))
        print ("Num of false label {}/{} - {:.4f}".format(num_false, n, num_false/n))

    def _load_data(
        self,
        data_folder,
        dataset_type,
        num_sample=None,
        batch_size=8,
        split_ratio=0.1,
    ):
        data_file = "qna_data/pre_data/{}/{}.csv".format(
            data_folder, dataset_type
        )

        df = pd.read_csv(data_file, index_col=None)

        # convert string list to int list
        str_list_columns = self.input_feature_cols
        for col in str_list_columns:
            df.loc[:, col] = df.loc[:, col].apply(literal_eval)

        self.batch_size = batch_size

        if not num_sample:
            num_sample = df.shape[0]

        if dataset_type == "train":
            df = df.iloc[:num_sample]
            train_df, dev_df = train_test_split(df, test_size=split_ratio, random_state=99)

            self.train_set =  tf.data.Dataset.from_tensor_slices(dict(train_df))
            self.train_set = self.train_set.shuffle(buffer_size=train_df.shape[0], seed=99)
            self.train_set = self.train_set.batch(batch_size, drop_remainder=True)

            self.dev_set = tf.data.Dataset.from_tensor_slices(dict(dev_df))
            self.dev_set = self.dev_set.batch(batch_size)

            self.train_size = train_df.shape[0]
            self.dev_size = dev_df.shape[0]

            self._summary_data(train_df, "train")
            self._summary_data(dev_df, "dev")

        elif dataset_type == "test":
            dataset = tf.data.Dataset.from_tensor_slices(dict(df))
            self.test_set = dataset.batch(batch_size)
            self.test_size = num_sample

            self._summary_data(df, "test")

if __name__ == "__main__":

    trainer = Trainer(
        lang="vi",
        data_folder="vi_normal_uncased",
        model_type="vi_attentive_reader",
        lr=1e-4,
        current_epoch=0,
        input_feature_cols=["question", "text"],
    )

    # trainer = Trainer(
    #     lang="en",
    #     data_folder="en_normal_uncasedl",
    #     model_type="en_bert_bidaf",
    #     lr=1e-5,
    #     current_epoch=9,
    # )

    mode = sys.argv[1]

    if mode == "check":
        trainer.sanity_check()

    elif mode == "train_dev":
        trainer.train_dev(epochs=50)

    elif mode == "train_full":
        trainer.train_full(epochs=20)

    elif mode == "eval":
        trainer.eval()
