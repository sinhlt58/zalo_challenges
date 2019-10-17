import copy
import random
import csv

import numpy as np
import tensorflow as tf

from vocab import VocabEntry
from utils import write_json_data, read_json_data, get_method_key

class Dataset:

    train_keys = ["train_questions", "train_paragraphs", "test_questions", "test_paragraphs"]

    def __init__(
        self,
        train_question_texts,
        train_paragraph_texts,
        train_labels,
        train_titles,
        test_question_texts,
        test_paragraph_texts,
        test_q_p_ids,
        method,
    ):
        self.train_question_texts = self._split(train_question_texts)
        self.train_paragraph_texts = self._split(train_paragraph_texts)
        self.train_labels = self._convert_labels(train_labels)
        self.train_titles = train_titles
        self.test_question_texts = self._split(test_question_texts)
        self.test_paragraph_texts = self._split(test_paragraph_texts)
        self.test_q_p_ids = test_q_p_ids

        self.method = method

    def _convert_labels(self, labels):
        return np.array([1 if l else 0 for l in labels]).reshape(len(labels), 1).astype(np.float32)

    def _split(self, texts):
        return [text.split(" ") for text in texts]

    def build_data(self):
        corpus = self.train_paragraph_texts + \
                 self.train_question_texts + \
                 self.test_question_texts + \
                 self.test_paragraph_texts

        vocab = VocabEntry.from_corpus(corpus, freq_cutoff=1)

        vocab_file = "qna_data/{}_vocab.json".format(self.method)
        vocab.save_json(vocab_file)

        self.train_questions = vocab.padd_sents(
            self.train_question_texts, start_end=False
        )
        self.train_paragraphs = vocab.padd_sents(
            self.train_paragraph_texts, start_end=False
        )
        self.test_questions = vocab.padd_sents(
            self.test_question_texts, start_end=False
        )
        self.test_paragraphs = vocab.padd_sents(
            self.test_paragraph_texts, start_end=False
        )

        save_data = {
            "train_questions": self.train_questions,
            "train_paragraphs": self.train_paragraphs,
            "test_questions": self.test_questions,
            "test_paragraphs": self.test_paragraphs,
        }

        save_file = "qna_data/{}_dataset.json".format(self.method)
        write_json_data(save_file, save_data)

        self.vocab = vocab

        self._to_numpy()

        print ("corpus len: ", len(corpus))
        print (corpus[0])
        print ("max length: ", vocab.max_sent_len)

    def summary(self):
        attributes = copy.deepcopy(self.train_keys)
        attributes.append("train_labels")

        for attribute in attributes:
            value = getattr(self, attribute)
            print ("{} length: {}".format(
                attribute, len(value)
            ))

    def get_sample(
        self, num_sample=200, split_ratio=0.1, shuffle=True, debug=True, full=False
    ):
        if full:
            return self.train_questions, self.train_paragraphs, self.train_labels, \
                   (self.test_questions, self.test_paragraphs)

        indexes = [i for i in range(0, len(self.train_questions))]
        random.shuffle(indexes)

        max_len = len(self.train_questions)
        if num_sample > max_len:
            num_sample = max_len

        indexes = indexes[:num_sample]

        split_idx = int(num_sample * split_ratio)

        dev_indexes = indexes[:split_idx]
        train_indexes = indexes[split_idx:]

        train_questions = self.train_questions[train_indexes]
        train_paragraphs = self.train_paragraphs[train_indexes]
        train_labels = self.train_labels[train_indexes]

        dev_questions = self.train_questions[dev_indexes]
        dev_paragraphs = self.train_paragraphs[dev_indexes]
        dev_labels = self.train_labels[dev_indexes]

        if debug:
            print ("train_questions length: ", len(train_questions))
            print ("train_paragraphs length: ", len(train_paragraphs))
            print ("train_labels length: ", len(train_labels))
            print ("dev_questions length: ", len(dev_questions))
            print ("dev_paragraphs length: ", len(dev_paragraphs))
            print ("dev_labels length: ", len(dev_labels))

            print ("Debug train_questions: ", self.vocab.listindices2words(train_questions[:1]))
            print ("Debug train_paragraphs: ", self.vocab.listindices2words(train_paragraphs[:1]))
            print ("Debug train_labels: ", train_labels[:1])
            print ("Debug dev_questions: ", self.vocab.listindices2words(dev_questions[:1]))
            print ("Debug dev_paragraphs: ", self.vocab.listindices2words(dev_paragraphs[:1]))
            print ("Debug dev_labels: ", dev_labels[:1])

        return train_questions, train_paragraphs, train_labels, \
               (dev_questions, dev_paragraphs, dev_labels)

    def _to_numpy(self):

        for key in self.train_keys:
            setattr(self, key, np.array(getattr(self, key)))

    def _from_json(self):
        vocab_file = "qna_data/{}_vocab.json".format(self.method)
        dataset_file = "qna_data/{}_dataset.json".format(self.method)

        self.vocab = VocabEntry.from_json(vocab_file)

        dataset_json = read_json_data(dataset_file)

        for key in self.train_keys:
            setattr(self, key, dataset_json[key])

        self._to_numpy()

    def get_test_data(self):
        return self.test_questions, self.test_paragraphs

    def gen_submit(self, y_preds):
        submit_file = 'qna_data/submit.csv'

        writer = csv.writer(open(submit_file, 'w', encoding='utf-8', newline=''))

        writer.writerow(["test_id", "answer"])

        for idx, y in enumerate(y_preds):
            if y:
                writer.writerow([self.test_q_p_ids[idx][0], self.test_q_p_ids[idx][1]])

        print ("Done write sumbit file to ", submit_file)

    @classmethod
    def from_json(cls, method="normal", mode="train"):
        dataset = cls.load_qna_data(method=method, build_data=False, mode=mode)
        dataset._from_json()

        return dataset

    @classmethod
    def load_qna_data(cls, method="normal", build_data=True, mode="train"):
        train_file = "qna_data/{}_train.json".format(method)
        test_file = "qna_data/{}_test.json".format(method)

        train_question_texts, train_paragraph_texts = [], []
        train_labels = []
        train_titles = []
        test_question_texts, test_paragraph_texts = [], []
        test_q_p_ids = []

        test_json = read_json_data(test_file)

        pre_title_key = get_method_key("title", method)
        pre_question_key = get_method_key("question", method)
        pre_text_key = get_method_key("text", method)
        pre_paragraphs_key = get_method_key("paragraphs", method)

        if mode == "train":
            train_json = read_json_data(train_file)
            for train_sample in train_json:
                train_question_texts.append(train_sample[pre_question_key])
                train_paragraph_texts.append(train_sample[pre_text_key])
                train_labels.append(train_sample["label"])
                train_titles.append(train_sample[pre_title_key])

        for test_sample in test_json:
            for p in test_sample[pre_paragraphs_key]:
                test_question_texts.append(test_sample[pre_question_key])
                test_paragraph_texts.append(p["text"])
                test_q_p_ids.append((test_sample["__id__"], p["id"]))

        dataset = cls(
            train_question_texts=train_question_texts,
            train_paragraph_texts=train_paragraph_texts,
            train_labels=train_labels,
            train_titles=train_titles,
            test_question_texts=test_question_texts,
            test_paragraph_texts=test_paragraph_texts,
            test_q_p_ids=test_q_p_ids,
            method=method,
        )

        if build_data:
            dataset.build_data()

        return dataset


if __name__ == "__main__":

    dataset = Dataset.load_qna_data(method="normal")
    dataset.summary()

    print ("train_question_texts: ", dataset.train_question_texts[:5])
    print ("train_paragraph_texts: ", dataset.train_paragraph_texts[:5])
    print ("train_labels: ", dataset.train_labels[:5])
    print ("train_titles: ", dataset.train_titles[:5])
    print ("test_question_texts: ", dataset.test_question_texts[:5])
    print ("test_paragraph_texts: ", dataset.test_paragraph_texts[:5])
    print ("test_q_p_ids: ", dataset.test_q_p_ids[:5])

    print ("train_questions: ", dataset.train_questions[:5])
    print ("train_paragraphs: ", dataset.train_paragraphs[:5])
    print ("test_questions: ", dataset.test_questions[:5])
    print ("test_paragraphs: ", dataset.test_paragraphs[:5])
