import csv
import re
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import read_json_data, create_folder

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
    print ("Write file {}".format(tsv_file))

def _write_pids(id_pids, lang, dataset_type, pids_file=None):
    if pids_file is None:
        pids_file = "qna_data/glue_data/{}/final/{}_pids.txt".format(lang, dataset_type)
    with open(pids_file, "w") as f:
        for pid in id_pids:
            f.write("{}\n".format(pid))
    print ("Write file {}".format(pids_file))

def _count_word(txt):
    return len(txt.split())

def _summary_df(df):
    total = int(df.shape[0])
    num_has_answer = int((df["label"] == "entailment").sum())
    num_no_answer = total - num_has_answer
    print("total", total)
    print("num_has_answer", num_has_answer)
    print("num_no_answer", num_no_answer)
    print("percentage_has_answer", num_has_answer / total)
    print("percentage_no_answer", num_no_answer / total)
    print("has_answer_ratio", num_has_answer / num_no_answer)


def _split_k_folds(df, lang, num_folds=5, batch_size=32, max_len=512):
    from sklearn.utils import shuffle
    from sklearn.model_selection import KFold

    n = df.shape[0]
    # we remove number of batch_remainder longest sequences
    examples_remain = n % batch_size
    num_batch = n // batch_size

    # remain batch fold
    num_batch_remain = num_batch % num_folds

    # 1. Remove examples_remain longest examples
    q_s_lengths = []
    for i, row in df.iterrows():
        len_q = _count_word(row["question"])
        len_s = _count_word(row["sentence"])
        q_s_lengths.append(len_q + len_s)
    df["q_s_len"] = q_s_lengths
    df = df.sort_values(by=["q_s_len"], ascending=False)
    df_to_remove = df.iloc[:examples_remain]
    df = df[examples_remain:]

    train_full_path = "qna_data/glue_data/{}/cv/train_zalo_full.tsv".format(lang)
    df = df.sort_values(by=["q_s_len"])
    _write_tsv(df, train_full_path)

    # 2. K folds split
    # add add remain examples to the train folds
    num_batch_remain_example = num_batch_remain * batch_size
    df = shuffle(df, random_state=16)
    # reset index
    df.reset_index(inplace=True, drop=True)
    del df["q_s_len"]

    # add later df
    add_later_df = df[:num_batch_remain_example]
    df = df[num_batch_remain_example:]

    print ("num examples: ", n)
    print ("batch size: ", batch_size)
    print ("remove examples: ", examples_remain)
    print ("num batch size: ", num_batch)
    print ("num folds: ", num_folds)
    print ("num batch remain: ", num_batch_remain)
    print ("num batch remain example: ", num_batch_remain_example)
    kf = KFold(n_splits=num_folds)
    fold_i = 0
    for train_index, dev_index in kf.split(df):
        train_df = df.loc[train_index, :]
        dev_df = df.loc[dev_index, :]

        train_df = pd.concat([train_df, add_later_df])

        print ("********************** Summary fold {} **********************".format(fold_i))
        print ("-----Summary train_df-----")
        _summary_df(train_df)
        print ("-----Summary dev_df-----")
        _summary_df(dev_df)

        fold_folder = "qna_data/glue_data/{}/cv/fold{}".format(lang, fold_i)
        create_folder(fold_folder)

        train_df_path = "{}/train.tsv".format(fold_folder)
        dev_df_path = "{}/dev.tsv".format(fold_folder)
        dev_pids_path = "{}/dev_pids.txt".format(fold_folder)

        _write_tsv(train_df, train_df_path)
        _write_tsv(dev_df, dev_df_path)
        _write_pids(dev_df["pid"], lang, "dev", dev_pids_path)

        fold_i += 1


def _split_train(in_dir, df, train_out, dev_out):
    train_df, dev_df = train_test_split(df, test_size=0.18, random_state=100)

    train_out_path = "{}/{}.tsv".format(in_dir, train_out)
    dev_out_path = "{}/{}.tsv".format(in_dir, dev_out)

    _write_tsv(train_df, train_out_path)
    _write_tsv(dev_df, dev_out_path)
    _write_pids(dev_df["index"], "vi", dev_out)


def _zalo_to_glue(file_name, pre_method="normal_cased"):
    parts = file_name.split("_")
    lang = parts[0]
    dataset_type = parts[1]
    json_file = "qna_data/{}.json".format(file_name)
    tsv_file = "qna_data/glue_data/{}/final/{}.tsv".format(lang, dataset_type)

    json_samples = read_json_data(json_file)

    glue_dict = {
        "index": [],
        "question": [],
        "sentence": [],
        "label": [],
        "pid": [],
    }
    id_pids = []

    for idx, json_sample in enumerate(json_samples):
        glue_dict["index"].append(idx)
        glue_dict["question"].append(
            _pre_process_question(json_sample["{}_question".format(pre_method)])
        )
        glue_dict["sentence"].append(
            _pre_process_sentence(json_sample["{}_text".format(pre_method)])
        )
        glue_dict["label"].append(
            "entailment" if json_sample["label"] else "not_entailment"
        )
        id_pid = "{}@{}".format(json_sample["id"], json_sample["pid"])
        id_pids.append(id_pid)
        glue_dict["pid"].append(id_pid)

    glue_df = pd.DataFrame(glue_dict)

    if file_name == "vi_train":
        _split_train("qna_data/glue_data/vi/final", glue_df, "train90", "dev10")

    if file_name == "vi_btrain":
        _split_train("qna_data/glue_data/vi/final", glue_df, "btrain90", "bdev10")

    # divide in into k folds here
    # _split_k_folds(glue_df, lang)


    if "test" in file_name or "private" in file_name:
        _write_pids(id_pids, lang, dataset_type)

    _write_tsv(glue_df, tsv_file)


def convert_zalo_to_glue():
    file_names = ["vi_train", "vi_test", "vi_private", "vi_squad", "vi_btrain", "vi_bsquad", "vi_ltest"]
    # file_names = ["vi_train"]
    # file_names = ["vi_test"]
    # file_names = ["vi_squad"]
    # file_names = ["vi_ltest"]

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

def concat_tsv_files(in_dir, in_filenames, out_filename):
    paths = []
    for fn in in_filenames:
        paths.append("{}/{}.tsv".format(in_dir, fn))

    out_path = "{}/{}.tsv".format(in_dir, out_filename)
    _concat_tsv_files(paths, out_path)


def replace_question(in_dir, from_f, to_f):
    from_path = "{}/{}.tsv".format(in_dir, from_f)
    to_path = "{}/{}.tsv".format(in_dir, to_f)

    from_df = pd.read_csv(from_path, encoding="utf-8", sep="\t", quoting=csv.QUOTE_NONE)
    to_df = pd.read_csv(to_path, encoding="utf-8", sep="\t", quoting=csv.QUOTE_NONE)

    to_df["question"] = from_df["question"]

    _write_tsv(to_df, to_path)

if __name__ == "__main__":
    data_task_type = sys.argv[1]
    in_dir = "qna_data/glue_data/vi/final"

    if data_task_type == "zalo":
        convert_zalo_to_glue()
    elif data_task_type == "squad":
        convert_squad_to_glue()
    elif data_task_type == "concat":
        # concat_tsv_files(in_dir, ["train90", "squad"], "train90_squad")
        # concat_tsv_files(in_dir, ["train90", "squad", "btrain90", "bsquad"], "btrain90_squad")
        # concat_tsv_files(in_dir, ["train", "squad", "btrain", "bsquad"], "btrain100_squad")
        concat_tsv_files(in_dir, ["train", "squad"], "train100_squad")
    elif data_task_type == "replace_question":
        replace_question(in_dir, "train", "btrain")
        replace_question(in_dir, "train90", "btrain90")
        replace_question(in_dir, "dev10", "bdev10")
        replace_question(in_dir, "squad", "bsquad")
