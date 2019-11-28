
import sys
import pandas as pd
import csv

from utils import read_json_data, write_json_data

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

def _vi_to_zalo():
    squad_dir = "squad_data"
    zalo_samples = []
    _id = 0
    for file_name in ["vi_train-v2.0.json", "vi_dev-v2.0.json"]:
        file_path = "{}/{}".format(squad_dir, file_name)
        samples = read_json_data(file_path)
        for sample in samples["data"]:
            title = sample["title"]
            for p in sample["paragraphs"]:
                context = p["context"]
                for qa in p["qas"]:
                    zalo_sample = {
                        "id": "squad-{}".format(_id),
                        "title": title,
                        "question": qa["question"],
                        "text": context,
                        "label": not qa["is_impossible"],
                    }
                    zalo_samples.append(zalo_sample)
                    _id += 1

    out_path = "qna_data/squad.json"
    write_json_data(out_path, zalo_samples)
    print ("Write file {}".format(out_path))

if __name__ == "__main__":
    arg1 = sys.argv[1]

    if arg1 == "vi_to_qnli":
        _vi_to_qnli("train")
        _vi_to_qnli("dev")
    elif arg1 == "vi_to_zalo":
        _vi_to_zalo()
    else:
        pass
