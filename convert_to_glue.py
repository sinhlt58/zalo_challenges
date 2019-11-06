import csv
import re
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import read_json_data

def _pre_process_common(text):
    text = text.replace("\\t", " ")

    tokens = text.split()
    text = " ".join(tokens)

    return text

def _pre_process_question(text):
    text = _pre_process_common(text)
    return text

def _pre_process_sentence(text):
    text = _pre_process_common(text)
    return text

def _write_tsv(df, tsv_file):
    df.to_csv(tsv_file, encoding="utf-8", quoting=csv.QUOTE_NONE, sep="\t", index=False,
                columns=["index", "question", "sentence", "label"])

def _zalo_to_glue(file_name, split_ratio=0.2):
    parts = file_name.split("_")
    lang = parts[0]
    dataset_type = parts[1]
    json_file = "qna_data/{}.json".format(file_name)
    tsv_file = "qna_data/glue_data/{}/{}.tsv".format(lang, dataset_type)

    json_samples = read_json_data(json_file)

    glue_dict = {
        "index": [],
        "question": [],
        "sentence": [],
        "label": [],
    }
    id_pids = []

    for idx, json_sample in enumerate(json_samples):
        glue_dict["index"].append(idx)
        glue_dict["question"].append(
            _pre_process_question(json_sample["question"])
        )
        glue_dict["sentence"].append(
            _pre_process_sentence(json_sample["text"])
        )
        glue_dict["label"].append(
            "entailment" if json_sample["label"] else "not_entailment"
        )
        id_pids.append("{}-{}".format(json_sample["id"], json_sample["pid"]))

    glue_df = pd.DataFrame(glue_dict)

    if "train" in file_name:
        glue_df, dev_df = train_test_split(glue_df, test_size=split_ratio, random_state=99)
        dev_tsv_file = "qna_data/glue_data/{}/{}.tsv".format(lang, "dev")
        _write_tsv(dev_df, dev_tsv_file)

    if "test" in file_name or "private" in file_name:
        pids_file = "qna_data/glue_data/{}/{}_pids.txt".format(lang, dataset_type)
        with open(pids_file, "w") as f:
            for pid in id_pids:
                f.write("{}\n".format(pid))

    _write_tsv(glue_df, tsv_file)

def convert_zalo_to_glue():
    file_names = ["en_train", "en_test", "vi_train", "vi_test"]

    for file_name in file_names:
        _zalo_to_glue(file_name)

def _squad_to_examples(squad_json_file, start_id=0):
    json_data = read_json_data(squad_json_file)

    ids = []
    questions = []
    sentences = []
    labels = []
    for json_sample in json_data["data"]:
        for paragraph in json_sample["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                ids.append(start_id)
                questions.append(_pre_process_question(qa["question"]))
                sentences.append(_pre_process_sentence(context))
                labels.append("not_entailment" if qa["is_impossible"] else "entailment")

                start_id += 1

    return ids, questions, sentences, labels

def convert_squad_to_glue():
    train_json_file = "squad_data/train-v2.0.json"
    dev_json_file = "squad_data/dev-v2.0.json"

    glue_dict = {
        "index": [],
        "question": [],
        "sentence": [],
        "label": [],
    }
    train_ids, train_questions, train_sentences, train_labels = _squad_to_examples(train_json_file, start_id=0)
    dev_ids, dev_questions, dev_sentences, dev_labels = _squad_to_examples(dev_json_file, start_id=len(train_ids))

    glue_dict["index"] += train_ids + dev_ids
    glue_dict["question"] += train_questions + dev_questions
    glue_dict["sentence"] += train_sentences + dev_sentences
    glue_dict["label"] += train_labels + dev_labels

    glue_df = pd.DataFrame(glue_dict)
    tsv_file = "qna_data/glue_data/en/squad_train.tsv"
    _write_tsv(glue_df, tsv_file)

def _is_train_dev(file1, file2):
    parts1 = file1.split('/')
    parts2 = file2.split('/')

    return (parts1[-1], parts2[-1]) == ('train.tsv', 'dev.tsv') or \
           (parts2[-1], parts1[-1]) == ('train.tsv', 'dev.tsv')

def _concat_tsv_files(file_names, merged_file):
    # combined_df = pd.concat([pd.read_csv(f) for f in file_names])
    dfs = []
    start_id = 0
    for i, f in enumerate(file_names):
        df = pd.read_csv(f, encoding="utf-8", quoting=csv.QUOTE_NONE, sep="\t")
        df["index"] = df["index"].apply(lambda a: a + start_id)
        dfs.append(df)
        # skip plus start_id for pair train, dev
        if i < len(file_names) - 1 and _is_train_dev(f, file_names[i+1]):
            continue
        start_id += df.shape[0]
        print ("start_id: ", start_id)

    combined_df = pd.concat(dfs)

    _write_tsv(combined_df, merged_file)

def concat_tsv_files():
    folder = "qna_data/glue_data/en"
    train = "{}/train.tsv".format(folder)
    dev = "{}/dev.tsv".format(folder)
    squad_train = "{}/squad_train.tsv".format(folder)

    zalo_full_train = "{}/zalo_full_train.tsv".format(folder)
    squad_zalo_train = "{}/squad_zalo_train.tsv".format(folder)
    squad_zalo_full_train = "{}/squad_zalo_full_train.tsv".format(folder)

    _concat_tsv_files([train, dev], zalo_full_train)
    _concat_tsv_files([train, squad_train], squad_zalo_train)
    _concat_tsv_files([squad_train, zalo_full_train], squad_zalo_full_train)

if __name__ == "__main__":
    data_task_type = sys.argv[1]

    if data_task_type == "zalo":
        convert_zalo_to_glue()
    elif data_task_type == "squad":
        convert_squad_to_glue()
    elif data_task_type == "concat":
        concat_tsv_files()
