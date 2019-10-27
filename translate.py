import sys
import time

from textblob import TextBlob
from textblob.exceptions import NotTranslated
from utils import read_json_data, write_json_data, get_method_key

def translate(dataset_types):
    for dataset_type in dataset_types:
        vi_file = "qna_data/raw_vi_{}.txt".format(dataset_type)
        en_file = "qna_data/raw_en_{}.txt".format(dataset_type)

        current_line = sum(1 for line in open(en_file, "r", encoding="utf-8"))
        print ("{}. current_line: {}".format(dataset_type, current_line))

        lines = [l.strip() for l in open(vi_file, "r", encoding="utf-8")]

        with open(en_file, "a", encoding="utf-8") as f:
            for idx in range(current_line, len(lines)):
                source_line = lines[idx]
                try:
                    target_line = str(TextBlob(source_line).translate("vi", "en"))
                    f.write("{}\n".format(target_line))
                except NotTranslated:
                    print ("{}. Line {}. Unchange after translate so write the source line".format(
                        dataset_type, idx
                    ))
                    f.write("{}\n".format(source_line))
                sleep = 0.5
                print ("{}. Translated line {}. Sleep for {}s ...".format(dataset_type, idx, sleep))
                time.sleep(sleep)


if __name__ == "__main__":
    dataset_types = sys.argv[1]

    if dataset_types == "all":
        dataset_types = ["train", "test"]
    else:
        dataset_types = [dataset_types]

    translate(dataset_types)
