import argparse
import os
import glob
import time

from textblob import TextBlob
from textblob.exceptions import NotTranslated

def translate(args, file_name):
    src_file = os.path.join(args.in_dir, file_name)
    trg_file = os.path.join(args.out_dir, file_name)

    if not os.path.exists(trg_file):
        print ("Create file {}".format(trg_file))
        open(trg_file, 'a', encoding="utf-8").close()

    current_line = sum(1 for line in open(trg_file, "r", encoding="utf-8"))
    print ("{}. current_line: {}".format(file_name, current_line))

    lines = [l.strip() for l in open(src_file, "r", encoding="utf-8")]

    if current_line == len(lines):
        print ("File {} already translated all sentences!".format(file_name))
        return

    with open(trg_file, "a", encoding="utf-8") as f:
        for idx in range(current_line, len(lines)):
            source_line = lines[idx]
            try:
                target_line = str(TextBlob(source_line).translate(args.src, args.trg))
                f.write("{}\n".format(target_line))
            except NotTranslated:
                print ("{}. Line {}. Unchange after translate so write the source line".format(
                    file_name, idx
                ))
                f.write("{}\n".format(source_line))
            sleep = 0.4
            print ("{}. Translated line {}. Sleep for {}s ...".format(file_name, idx, sleep))
            time.sleep(sleep)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_remote", action="store_true")
    parser.add_argument("--in_dir", default="glue_data/qnli/en", type=str, required=False)
    parser.add_argument("--out_dir", default="glue_data/qnli/vi", type=str, required=False)
    parser.add_argument("--src", default="en", type=str, required=False)
    parser.add_argument("--trg", default="vi", type=str, required=False)

    args = parser.parse_args()

    input_files = [
        "train_questions_raw_52332_52334.txt",
        "dev_questions_raw_0_4100.txt",
        "dev_sentences_raw_0_3911.txt",
    ]

    if args.do_remote:
        input_files = []
        for file_path in glob.glob("{}/*raw*.txt".format(args.in_dir)):
            file_name = file_path.split("/")[-1]
            input_files.append(file_name)

    for file_name in input_files:
        translate(args, file_name)
