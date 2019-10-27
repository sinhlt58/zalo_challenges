
import re
import sys
import csv
import string

from nltk.tokenize import word_tokenize, RegexpTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

from bert_official.tokenization import FullTokenizer
from vocab import VocabEntry
from utils import read_json_data, write_json_data, get_method_key, get_bert_paths, create_folder


class ViPreProcessor:

    def __init__(self, build_vocab, for_train, local_test_split=0.0):
        self.build_vocab = build_vocab
        self.for_train = for_train
        self.regex_tokenizer = RegexpTokenizer(r'\w+')

        self.question_size = 50
        self.text_size = 300

    def preprocess_qna_data(
        self, method, cased, dataset_types,
    ):
        folder_name = "{}_{}".format(method, cased)
        folder_path = "qna_data/pre_data/vi_{}".format(folder_name)
        create_folder(folder_path)

        # preprocess fields
        dataset_features_columns = {}
        for dataset_type in dataset_types:
            data_file = "qna_data/vi_{}.json".format(dataset_type)

            # Init features columns
            if self.for_train:
                features_columns = {
                    "id": [],
                    "question": [],
                    "text": [],
                    "label": [],
                    "pid": [],
                }

            json_samples = read_json_data(data_file)

            for json_sample in json_samples:

                if self.for_train:
                    features_columns["id"].append(json_sample["id"])
                    features_columns["label"].append(1 if json_sample["label"] else 0)
                    features_columns["pid"].append(json_sample["pid"])

                for key in ["question", "text"]:
                    pre_key = "{}_{}_{}".format(
                        method, cased, key
                    )
                    pre_text, tokens = self.pre_process_text(
                        json_sample[key], method, cased, self.for_train
                    )
                    json_sample[pre_key] = pre_text

                    if self.for_train:
                        features_columns[key].append(tokens)

            # samples with preprocessed keys
            write_json_data(data_file, json_samples)
            print ("{}. Length {}. Done write to file {}".format(
                dataset_type, len(json_samples), data_file
            ))

            # save for writing later when we have vocab
            if self.for_train:
                dataset_features_columns[dataset_type] = features_columns

        # build vocab
        vocab_file = "{}/vocab.json".format(folder_path, dataset_type)
        if self.build_vocab:
            self._build_vocab(vocab_file, method, cased)
        else:
            self.vocab = VocabEntry.from_json(vocab_file)

        # write configs
        configs = {
            "vocab_size": len(self.vocab),
            "question_size": self.question_size,
            "text_size": self.text_size,
        }
        configs_file = "{}/configs.json".format(folder_path)
        write_json_data(configs_file, configs)
        print ("Done wirte config file {}".format(configs_file))

        # write features columns
        # generate featured dataset
        if self.for_train:
            for dataset_type, features_columns in dataset_features_columns.items():
                self.write_features_columns(
                    features_columns, folder_name, dataset_type
                )


    def _build_vocab(self, vocab_file, method, cased):
        corpus = []

        for dataset_type in ["train", "test"]:
            data_file = "qna_data/vi_{}.json".format(dataset_type)

            json_samples = read_json_data(data_file)

            for json_sample in json_samples:
                for key in ["question", "text"]:
                    pre_key = "{}_{}_{}".format(
                        method, cased, key
                    )
                    pre_text = json_sample[pre_key]

                    corpus.append(pre_text.split())

        self.vocab = VocabEntry.from_corpus(corpus, freq_cutoff=3)
        self.vocab.save_json(vocab_file)
        print ("Save vocab to file {}".format(vocab_file))

    def pre_process_text(self, text, method, cased, for_train):
        # remove references in wikipedia articles (ex: [1])
        text = re.sub(r'\[[0-9]+\]', "", text)

        # simple tokenize
        tokens = word_tokenize(text)
        text = " ".join(tokens)

        # remove punctuation
        punctuations = string.punctuation
        punctuations = re.sub(r'\(|\)|-', "", punctuations) # we keep ( )
        text = text.translate(str.maketrans('', '', punctuations))
        text = text.replace("()", "") # remove empty ()
        text = text.replace("( )", "") # remove ( )
        text = re.sub(r'\( .{1,2} \)', "", text) # remove stuff
        text = text.replace("–", " ")
        text = text.replace("-", " ")

        # replace number, entities
        text = re.sub(r'\b[0-9]+\b', "_NUMBER_", text)

        # remove spaces
        tokens = text.split()
        text = " ".join(tokens)

        if cased == "uncased":
            text = text.lower()

        tokens = text.split()

        return text, tokens

    def write_features_columns(self, features_columns, folder_name, dataset_type):
        folder_path = "qna_data/pre_data/vi_{}".format(folder_name)
        dataset_file = "{}/{}.csv".format(folder_path, dataset_type)
        create_folder(folder_path)

        fc_df = pd.DataFrame(features_columns)

        fc_df["question"] = self.vocab.padd_sents(fc_df["question"])
        fc_df["text"] = self.vocab.padd_sents(fc_df["text"])

        fc_df["question"] = fc_df["question"].apply(lambda q: q[:self.question_size])
        fc_df["text"] = fc_df["text"].apply(lambda t: t[:self.text_size])

        num_true = (fc_df["label"] == 1).sum()
        num_false = (fc_df["label"] == 0).sum()

        print ("{} num_true: {}".format(dataset_type, num_true))
        print ("{} num_false: {}".format(dataset_type, num_false))

        fc_df.to_csv(dataset_file, encoding="utf-8", index=False,
                     columns=["id", "pid", "label", "question", "text"])

        print ("Done write to file {}".format(dataset_file))


class EnPreProcessor:

    def __init__(self, bert_type, for_train):
        bert_model_path, bert_vocab_file, do_lower_case\
            = get_bert_paths(bert_type)
        self.bert_tokenizer = FullTokenizer(
            vocab_file=bert_vocab_file,
            do_lower_case=do_lower_case,
        )
        self.start_id = self.bert_tokenizer.token_to_id("[CLS]")
        self.sep_id = self.bert_tokenizer.token_to_id("[SEP]")
        self.padd_id = self.bert_tokenizer.token_to_id("[PAD]")

        self.question_size = 50
        self.text_size = 250
        self.max_length = 300

        self.for_train = for_train

    def preprocess_qna_data(
        self, method, bert_type, dataset_types,
    ):
        for dataset_type in dataset_types:
            data_file = "qna_data/en_{}.json".format(dataset_type)

            # Init features columns
            if self.for_train:
                features_columns = {
                    "id": [],
                    "question": [],
                    "text": [],
                    "label": [],
                    "pid": [],
                }

            json_samples = read_json_data(data_file)

            for json_sample in json_samples:
                if self.for_train:
                    features_columns["id"].append(json_sample["id"])
                    features_columns["label"].append(1 if json_sample["label"] else 0)
                    features_columns["pid"].append(json_sample["pid"])

                for key in ["question", "text"]:
                    pre_key = "{}_{}_{}".format(
                        method, bert_type, key
                    )
                    pre_text, tokens_id = self.pre_process_text(
                        json_sample[key], method, self.for_train
                    )
                    json_sample[pre_key] = pre_text

                    if self.for_train:
                        features_columns[key].append(tokens_id)

            # samples with preprocessed keys
            write_json_data(data_file, json_samples)
            print ("{}. Length {}. Done write to file {}".format(
                dataset_type, len(json_samples), data_file
            ))

            # generate featured dataset
            if self.for_train:
                folder_name = "{}_{}".format(method, bert_type)
                self.write_features_columns(
                    features_columns, folder_name, dataset_type
                )

    def pre_process_text(self, text, method, for_train):
        tokens = self.bert_tokenizer.tokenize(text)
        text = " ".join(tokens)

        tokens_id = []
        if for_train:
            tokens_id = self.bert_tokenizer.convert_tokens_to_ids(tokens)

        return text, tokens_id

    def write_features_columns(self, features_columns, folder_name, dataset_type):
        folder_path = "qna_data/pre_data/en_{}".format(folder_name)
        dataset_file = "{}/{}.csv".format(folder_path, dataset_type)
        create_folder(folder_path)

        # padd question and text
        # questions = features_columns["question"]
        # texts = features_columns["text"]

        # features_columns["question"] = pad_sequences(
        #     questions, maxlen=self.question_size, value=self.padd_id, padding="post"
        # )
        # features_columns["text"] = pad_sequences(
        #     texts, maxlen=self.text_size, value=self.padd_id, padding="post"
        # )

        # pack question and text together for BERT input
        csv_f = csv.writer(open(dataset_file, "w", encoding="utf-8", newline=""))
        csv_f.writerow(["id", "pid", "label", "input_features"])

        for i, question in enumerate(features_columns["question"]):
            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0     0   0   0  0     0 0

            type_ids = [1] * self.max_length
            text = features_columns["text"][i]

            # concatenate the question ids
            question_text = [self.start_id] + question[:self.question_size] + [self.sep_id]
            for idx, _ in enumerate(question_text):
                type_ids[idx] = 0 # mask 0 for the first sequence

            # concatenate the text ids
            question_text = question_text + \
                text[:(self.max_length - len(question_text) - 1)] + \
                [self.sep_id]

            # padd the remaining
            question_text = question_text + [self.padd_id] * (self.max_length - len(question_text))

            input_features = [question_text, type_ids]

            csv_f.writerow([
                features_columns["id"][i],
                features_columns["pid"][i],
                features_columns["label"][i],
                input_features,
            ])

        print ("Done write to file {}".format(dataset_file))


def preprocess_qna_data(config):
    language = config["language"]

    if language == "vi":
        for method in config["methods"]:
            for cased in config["cases"]:
                processor = ViPreProcessor(
                    config["build_vocab"],
                    config["for_train"],
                    config["local_test_split"],
                )
                processor.preprocess_qna_data(
                    method,
                    cased,
                    config["dataset_types"],
                )

    elif language == "en":
        for method in config["methods"]:
            for bert_type in config["bert_types"]:
                processor = EnPreProcessor(
                    bert_type, config["for_train"],
                )
                processor.preprocess_qna_data(
                    method,
                    bert_type,
                    config["dataset_types"],
                )


if __name__ == "__main__":

    configs = [
        {
            "language": "vi",
            "dataset_types": ["train", "test"],
            "methods": ["normal"],
            "cases": ["cased", "uncased"],
            "local_test_split": 0.2,
            "build_vocab": True,
            "for_train": True, # Will build ready datasets for training
        },
        # {
        #     "language": "en",
        #     "dataset_types": ["train", "test"],
        #     "methods": ["normal"],
        #     "bert_types": ["uncasedl"],
        #     "for_train": True, # Will build ready datasets for training
        # }
    ]

    for config in configs:
        preprocess_qna_data(config)
