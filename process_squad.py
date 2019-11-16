
import sys
import pandas as pd
import csv

from utils import read_json_data

def _write_tsv(df, tsv_file):
    df.to_csv(tsv_file, encoding="utf-8", quoting=csv.QUOTE_NONE, sep="\t", index=False,
                columns=["index", "question", "sentence", "label"])

def _simple_preprocess(text):
    tokens = text.split()
    return " ".join(tokens).strip()

def _vi_to_qnli(dataset, lang="vi"):
    squad_data_folder = "squad_data"
    json_path = "{}/{}_{}-v2.0.json".format(squad_data_folder, lang, dataset)

    json_examples = read_json_data(json_path)

    id_idx = 0
    df_dict = {
        "index": [],
        "question": [],
        "sentence": [],
        "label": [],
    }
    for json_example in json_examples["data"]:
        for p in json_example["paragraphs"]:
            context = p["context"]
            for qa in p["qas"]:
                df_dict["index"].append(id_idx)
                df_dict["question"].append(_simple_preprocess(qa["question"]))
                df_dict["sentence"].append(_simple_preprocess(context))
                df_dict["label"].append("not_entailment" if qa["is_impossible"] else "entailment")
                id_idx += 1

    df = pd.DataFrame(df_dict)
    tsv_path = "{}/{}_{}.tsv".format(squad_data_folder, lang, dataset)
    _write_tsv(df, tsv_path)
    print ("Write file {}".format(tsv_path))


if __name__ == "__main__":
    arg1 = sys.argv[1]

    if arg1 == "vi_to_qnli":
        _vi_to_qnli("train")
        _vi_to_qnli("dev")
    else:
        pass
