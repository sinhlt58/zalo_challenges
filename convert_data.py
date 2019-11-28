import sys
import copy

from utils import read_json_data, write_json_data, get_method_key

def _remove_break_lines(text):
    text = text.strip().replace("\n", " ").replace("\r", "")
    text = " ".join(text.split())
    return text

def write_txt_for_translation(json_samples, dataset_type):
    raw_vi_file = "qna_data/raw_vi_{}.txt".format(dataset_type)
    raw_id_type_file = "qna_data/raw_id_type_{}.txt".format(dataset_type)

    with open(raw_vi_file, "w", encoding="utf-8") as raw_data:
        with open(raw_id_type_file, "w", encoding="utf-8") as raw_id_type:
            for json_sample in json_samples:
                _id = json_sample["id"]

                is_write_question = True
                if dataset_type in ["test", "private"]:
                    if json_sample["pid"] != "p1":
                        is_write_question = False

                if is_write_question:
                    question_raw = json_sample["question"]
                    id_question_type = "{}\tquestion".format(
                        _id
                    )
                    raw_id_type.write("{}\n".format(id_question_type))
                    raw_data.write("{}\n".format(_remove_break_lines(
                        question_raw
                    )))

                text_raw = json_sample["text"]
                id_text_type = "{}\ttext\t{}\t{}".format(
                    _id, json_sample["pid"], json_sample["label"]
                )
                raw_id_type.write("{}\n".format(id_text_type))
                raw_data.write("{}\n".format(_remove_break_lines(
                    text_raw
                )))


def convert_data(dataset_type, include_txt=True):
    file_path = "qna_data/{}.json".format(dataset_type)

    data_json = read_json_data(file_path)

    converted_samples = []

    for sample_json in data_json:
        converted_sample = None

        if dataset_type in ["train", "squad"]:
            converted_sample = sample_json
            converted_sample["pid"] = "p1"
            converted_samples.append(converted_sample)

        elif dataset_type in ["test", "private", "ltest"]:
            for p in sample_json["paragraphs"]:
                if 'label' in p:
                    label = True if p['label'] == '1' else False
                else:
                    label = False
                converted_sample = {
                    "id": sample_json["__id__"],
                    "title": sample_json["title"],
                    "question": sample_json["question"],
                    "text": p["text"],
                    "label": label,
                    "pid": p["id"]
                }
                converted_samples.append(converted_sample)

    new_file_path = "qna_data/vi_{}.json".format(dataset_type)

    write_json_data(new_file_path, converted_samples)
    print ("Length {}. Done write to file {}".format(len(converted_samples), new_file_path))

    write_txt_for_translation(converted_samples, dataset_type) # write only vi files
    print ("Done write raw files for translation")

def convert_raw_en_to_json(dataset_type):
    raw_id_type_file = "qna_data/back_tran/raw_id_type_{}.txt".format(dataset_type)
    raw_en_file = "qna_data/back_tran/raw_vi_{}.txt".format(dataset_type)
    en_file = "qna_data/vi_{}.json".format(dataset_type)
    # for getting the title only
    vi_json_file = "qna_data/vi_{}.json".format(dataset_type[1:])
    vi_json_samples = read_json_data(vi_json_file)

    en_json_samples = []
    id_lines = [line.strip() for line in open(raw_id_type_file, "r", encoding="utf-8")]
    en_lines = [line.strip() for line in open(raw_en_file, "r", encoding="utf-8")]

    current_question = None
    text_idx = 0
    for i, id_line in enumerate(id_lines):
        parts = id_line.split("\t")

        if parts[1] == "question":
            current_question = {
                "id": parts[0],
                "question": en_lines[i],
            }

        elif parts[1] == "text":
            en_json_sample = copy.deepcopy(current_question)
            en_json_sample["title"] = vi_json_samples[text_idx]["title"]
            en_json_sample["text"] = en_lines[i]
            en_json_sample["label"] = True if parts[3] == "True" else False
            en_json_sample["pid"] = parts[2]

            en_json_samples.append(en_json_sample)
            text_idx += 1

    write_json_data(en_file, en_json_samples)
    print ("{}. Length {}. Done write to file {}".format(
        dataset_type, len(en_json_samples), en_file
    ))

if __name__ == "__main__":
    dataset_type = sys.argv[1]
    dataset_types = []
    back = None
    if len(sys.argv) > 2:
        back = sys.argv[2]

    if dataset_type == "all":
        dataset_types = ["train", "test", "btrain", "bsquad"]
    else:
        dataset_types = [dataset_type]

    for dt in dataset_types:
        if back == "back":
            convert_raw_en_to_json(dt) # only for en raw
        else:
            convert_data(dt)
