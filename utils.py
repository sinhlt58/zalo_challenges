import io
import os
import json
import pickle
import csv

import tensorflow as tf
import numpy as np

def write_json_data(path, data):
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def read_json_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)

def dump_object(obj, path):
    pickle.dump(obj, open(path, 'wb'))

def get_method_key(key, method):
        return "{}_{}".format(method, key)

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

bert_models_map = {
    "caseds": "cased_L-12_H-768_A-12",
    "casedl": "cased_L-24_H-1024_A-16",
    "uncaseds": "uncased_L-12_H-768_A-12",
    "uncasedl": "uncased_L-24_H-1024_A-16",
}

def get_bert_paths(bert_type):
    model_folder = bert_models_map[bert_type]
    model_path = "D:/works/zalo_challenges/models/bert/{}".format(model_folder)
    vocab_file = "{}/vocab.txt".format(model_path)

    do_lower_case = bert_type[:2] == "un"

    return model_path, vocab_file, do_lower_case

def convert_df_str(df):
    str_df = df.select_dtypes([np.object])
    str_df = str_df.stack().str.decode('utf-8').unstack()

    for col in str_df:
        df[col] = str_df[col]


def _load_glue_dataset(file_path):
    dict_data = {
        "idx": [],
        "label": [],
        "question": [],
        "sentence": [],
    }
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for idx, row in enumerate(reader):
            dict_data["idx"].append(idx)
            dict_data["label"].append(row[2])
            dict_data["question"].append(row[0])
            dict_data["sentence"].append(row[1])
    tf_dataset = tf.data.Dataset.from_tensor_slices(dict_data)
    return tf_dataset, len(dict_data["label"])

def load_glue_data(folder_path, task="qnli"):
    train_set, train_len = _load_glue_dataset("{}/{}/train.tsv".format(folder_path, task))
    dev_set, dev_len = _load_glue_dataset("{}/{}/dev.tsv".format(folder_path, task))

    return (train_set, train_len), (dev_set, dev_len)
