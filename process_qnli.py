import sys
import pandas as pd
import glob
import csv

from utils import write_json_data, read_json_data

def _write_tsv(df, tsv_file):
    df.to_csv(tsv_file, encoding="utf-8", quoting=csv.QUOTE_NONE, sep="\t", index=False,
                columns=["index", "question", "sentence", "label"])

def _split_tsv(tsv_file, num_split_row=5000):
    tsv_path = "glue_data/qnli/{}.tsv".format(tsv_file)
    split_folder = "glue_data/qnli/split_tsv"

    df = pd.read_csv(tsv_path, encoding="utf-8", quoting=csv.QUOTE_NONE, sep="\t")

    for i in range(0, df.shape[0], num_split_row):
        sub_df = df.iloc[i:i+num_split_row, :]
        out_file_name = "{}_{}.tsv".format(tsv_file, i)
        out_path = "{}/{}".format(split_folder, out_file_name)
        _write_tsv(sub_df, out_path)
        print ("Write file {}".format(out_path))

def _tsv_to_zalo(df):
    cache_dict = {}
    json_examples = []

    for i in range(0, df.shape[0]):
        id_index = df["index"][i]
        question = df["question"][i]
        text = df["sentence"][i]
        label = True if df["label"][i] == "entailment" else False

        if question not in cache_dict:
            pid = "p1"
            cache_dict[question] = {
                "count": 1,
                "id_index": id_index
            }
        else:
            cache_dict[question]["count"] += 1
            pid = "p{}".format(cache_dict[question]["count"])

        json_example = {
            "title": "fake_title",
            "id": str(id_index),
            "question": question,
            "text": text,
            "pid": pid,
            "label": label,
        }
        json_examples.append(json_example)

    return json_examples

def _to_zalo():
    split_folder = "glue_data\qnli\split_tsv"
    en_folder_zalo = "glue_data/qnli/en"

    for tsv_path in glob.glob("{}\*.tsv".format(split_folder)):
        df = pd.read_csv(tsv_path, encoding="utf-8", quoting=csv.QUOTE_NONE, sep="\t")
        json_examples = _tsv_to_zalo(df)
        json_file_name = tsv_path.split('\\')[-1].replace(".tsv", ".json")
        json_path = "{}/{}".format(en_folder_zalo, json_file_name)
        write_json_data(json_path, json_examples)
        print ("Write to file {}".format(json_path))

def _build_data(dataset):
    # train_questions.json
    # train_sentences.json
    # train_table.tsv
    tsv_path = "glue_data/qnli/{}.tsv".format(dataset)
    table_path = "glue_data/qnli/{}_table.tsv".format(dataset)
    df = pd.read_csv(tsv_path, encoding="utf-8", quoting=csv.QUOTE_NONE, sep="\t")

    table = {
        "index": [],
        "question": [],
        "sentence": [],
        "label": [],
    }
    questions_to_id_dict = {}
    sentences_to_id_dict = {}
    curr_q_id = 0
    curr_s_id = 0
    for i in range(0, df.shape[0]):
        index = df["index"][i]
        question = df["question"][i]
        sentence = df["sentence"][i]
        label = df["label"][i]

        if question not in questions_to_id_dict:
            questions_to_id_dict[question] = curr_q_id
            curr_q_id += 1
        if sentence not in sentences_to_id_dict:
            sentences_to_id_dict[sentence] = curr_s_id
            curr_s_id += 1

        q_id = questions_to_id_dict[question]
        s_id = sentences_to_id_dict[sentence]

        table["index"].append(index)
        table["label"].append(label)
        table["question"].append(q_id)
        table["sentence"].append(s_id)

    id_to_qs_dict = {v: k for k, v in questions_to_id_dict.items()}
    id_to_ss_dict = {v: k for k, v in sentences_to_id_dict.items()}
    id_to_qs_path = "glue_data/qnli/{}_questions.json".format(dataset)
    id_to_ss_path = "glue_data/qnli/{}_sentences.json".format(dataset)

    write_json_data(id_to_qs_path, id_to_qs_dict)
    print ("Write file {}".format(id_to_qs_path))
    write_json_data(id_to_ss_path, id_to_ss_dict)
    print ("Write file {}".format(id_to_ss_path))

    table_df = pd.DataFrame(table)
    _write_tsv(table_df, table_path)
    print ("Write file {}".format(table_path))

def _to_translate(dataset, fold=1):
    # prepare for translation
    json_files = ["questions", "sentences"]

    for json_file in json_files:
        json_path = "glue_data/qnli/{}_{}.json".format(dataset, json_file)
        json_samples = read_json_data(json_path)
        n = len(json_samples)
        split_num = n // fold

        for i in range(0, n, split_num):
            split_above = i + split_num
            if split_above > n:
                split_above = n
            raw_text_path = "glue_data/qnli/en/{}_{}_raw_{}_{}.txt".format(
                dataset, json_file, i, split_above
            )
            ids_path = "glue_data/qnli/en/{}_{}_ids_{}_{}.txt".format(
                dataset, json_file, i, split_above
            )
            raw_f = open(raw_text_path, "w", encoding="utf-8")
            ids_f = open(ids_path, "w", encoding="utf-8")

            for j in range(i, split_above):
                raw_f.write("{}\n".format(json_samples[str(j)]))
                ids_f.write("{}\n".format(j))

            raw_f.close()
            ids_f.close()

def _from_translate_to_json(dataset, src_folder, trg_folder, json_name):
    tran_ids_dict = {}

    for tran_path in glob.glob("{}/{}_{}_raw*.txt".format(trg_folder, dataset, json_name)):
        parts_idx = tran_path.split("\\")[-1].replace(".txt", "").split("_")
        ids_file_name = "{}_{}_ids_{}_{}.txt".format(dataset, json_name, parts_idx[-2], parts_idx[-1])
        ids_path = "{}/{}".format(src_folder, ids_file_name)

        text_lines = [l.strip() for l in open(tran_path, "r", encoding="utf-8")]
        ids_lines = [l.strip() for l in open(ids_path, "r", encoding="utf-8")]

        if len(text_lines) != len(ids_lines):
            print ("Error inside _from_translate_to_json")
            raise Exception

        for i, text_line in enumerate(text_lines):
            tran_ids_dict[ids_lines[i]] = text_line

    return tran_ids_dict

def _translate_to_qnli(dataset):
    # vi_train_questions.json
    # vi_train_sentences.json
    # vi_train.tsv
    src_folder = "glue_data/qnli/en"
    trg_folder = "glue_data/qnli/vi"

    json_files = ["questions", "sentences"]
    tran_ids_to_questions_dict = _from_translate_to_json(dataset, src_folder, trg_folder, "questions")
    tran_ids_to_sentences_dict = _from_translate_to_json(dataset, src_folder, trg_folder, "sentences")

    # build translated table
    table_path = "glue_data/qnli/{}_table.tsv".format(dataset)
    df_table = pd.read_csv(table_path, encoding="utf-8", quoting=csv.QUOTE_NONE, sep="\t")
    tran_dict = {
        "index": [],
        "question": [],
        "sentence": [],
        "label": [],
    }
    for i in range(0, df_table.shape[0]):
        index_id = df_table["index"][i]
        label = df_table["label"][i]
        question = tran_ids_to_questions_dict[str(df_table["question"][i])]
        sentence = tran_ids_to_sentences_dict[str(df_table["sentence"][i])]

        # remove end question mark from question
        if question[-1] == "?":
            question = question[:-1].strip()

        tran_dict["index"].append(index_id)
        tran_dict["question"].append(question)
        tran_dict["sentence"].append(sentence)
        tran_dict["label"].append(label)

    tran_df = pd.DataFrame(tran_dict)
    tran_dataset_path = "glue_data/qnli/vi_{}.tsv".format(dataset)
    _write_tsv(tran_df, tran_dataset_path)
    print ("Write file {}".format(tran_dataset_path))

    # write translated questions, sentences json
    tran_ids_to_questions_path = "glue_data/qnli/vi_{}_questions.json".format(dataset)
    tran_ids_to_sentences_path = "glue_data/qnli/vi_{}_sentences.json".format(dataset)

    write_json_data(tran_ids_to_questions_path, tran_ids_to_questions_dict)
    print ("Write file {}".format(tran_ids_to_questions_path))
    write_json_data(tran_ids_to_sentences_path, tran_ids_to_sentences_dict)
    print ("Write file {}".format(tran_ids_to_sentences_path))

if __name__ == "__main__":
    arg1 = sys.argv[1]

    if arg1 == "split":
        _split_tsv("train")
        _split_tsv("dev")
    elif arg1 == "to_zalo":
        _to_zalo()
    elif arg1 == "build_data":
        _build_data("train")
        _build_data("dev")
    elif arg1 == "to_translate":
        _to_translate("train", 4)
        _to_translate("dev", 1)
    elif arg1 == "tran_to_qnli":
        _translate_to_qnli("train")
        _translate_to_qnli("dev")
