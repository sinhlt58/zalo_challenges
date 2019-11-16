
import sys
import csv
import glob
import re
from underthesea import ner
import pandas as pd
from nltk.util import ngrams

def _is_train_dev(file1, file2):
    parts1 = file1.split('/')
    parts2 = file2.split('/')

    return (parts1[-1], parts2[-1]) == ('train.tsv', 'dev.tsv') or \
           (parts2[-1], parts1[-1]) == ('train.tsv', 'dev.tsv')

def _write_csv(df, tsv_file):
    df.to_csv(tsv_file, encoding="utf-8", quoting=csv.QUOTE_NONE, sep="\t", index=False,
                columns=["index", "question"])

def _get_combined_df(file_names):
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

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def _sort_end_write_csv(df, path, sort_by=[]):
    df = df.sort_values(by=sort_by)
    _write_csv(df, path)
    print ("Write file {}".format(path))

def _process_question(text):
    res = ner(text)

    # we replace words with entity name
    res_text = ""
    for i, r in enumerate(res):
        entity = r[3]
        sub = r[0]
        if entity == "O":
            res_text += sub + " "
        else:
            if i == len(res) - 1 or i < len(res) - 1 and ( res[i+1][3] == "O" or \
               (entity[:2] == "B-" and len(res[i+1][3]) > 1 and res[i+1][3][:2] == "B-") or \
               (entity != res[i+1][3]) ):
               res_text += "_{}_".format(entity[2:]) + " "
            else: # skip
                pass

    return res_text.strip()

def _write_questions_summary(index_ids, question, prefix, out_dir, merged_name):
    question_df = pd.DataFrame({
        "index": index_ids,
        "question": question,
    })
    question_path = "{}/{}_{}.csv".format(out_dir, prefix, merged_name)
    _sort_end_write_csv(question_df, question_path, "question")

VIETNAMESE_CHARACTERS = " 0123456789aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ"
def summary_tsv_files(tsv_dir, file_names, out_dir, punc=False):
    file_paths = []
    merged_name = ""
    for file_name in file_names:
        file_paths.append("{}/{}.tsv".format(tsv_dir, file_name))
        merged_name += "{}@".format(file_name)
    merged_name = merged_name[:-1]

    merged_df = _get_combined_df(file_paths)


    if punc:
        # write all punctuation
        punc_dict = {}
        for q in merged_df["question"]:
            for c in q:
                if c not in VIETNAMESE_CHARACTERS:
                    if c not in punc_dict:
                        punc_dict[c] = 0
                    punc_dict[c] += 1
        df_punc_dict = {
            'punc': [],
            'count': [],
            'unicode': [],
        }
        punc_chain = ''
        for c, count in punc_dict.items():
            df_punc_dict["punc"].append(c)
            df_punc_dict["count"].append(count)
            df_punc_dict["unicode"].append(c.encode('utf-8'))
            punc_chain += c
        punc_chain = punc_chain.encode('utf-8')
        print ('punc_chain: ', punc_chain)

        df_punc = pd.DataFrame(df_punc_dict)
        df_punc = df_punc.sort_values(by="count", ascending=False)
        punc_path = "{}/punc/{}.csv".format(out_dir, merged_name)
        df_punc.to_csv(punc_path, index=False, sep="\t", quoting=csv.QUOTE_NONE)
        print ("Write file {}".format(punc_path))
    else:
        processed_questions = []
        index_ids = []

        for i in range(0, merged_df.shape[0]):
            index_ids.append(merged_df["index"][i])
            processed_questions.append(_process_question(merged_df["question"][i]))

        _write_questions_summary(index_ids, merged_df["question"], "question", out_dir, merged_name)
        _write_questions_summary(index_ids, processed_questions, "entityquestion", out_dir, merged_name)

def _count_text_contains(texts, sub):
    count = 0
    for t in texts:
        if (' ' + sub + ' ') in (' ' + t + ' '):
            count += 1
    return count

def _write_ngram(questions_tokens, aug_dir, file_name, questions, ngram_num=1):
    list_ngrams = []

    for tokens in questions_tokens:
        nrams = list(ngrams(tokens, ngram_num))
        list_ngrams += nrams

    ngram_count = {}
    for n in list_ngrams:
        if n not in ngram_count:
            ngram_count[n] = 0
        ngram_count[n] += 1

    df_dict = {
        "count": [],
        "ngram": [],
        "num_q_contain": [],
    }
    for n, c in ngram_count.items():
        ngram_str = " ".join(n)
        df_dict["ngram"].append(ngram_str)
        df_dict["count"].append(c)
        df_dict["num_q_contain"].append(_count_text_contains(questions, ngram_str))

    out_path = "{}/ngram{}_{}.csv".format(aug_dir, ngram_num, file_name)
    df = pd.DataFrame(df_dict)
    df = df.sort_values(by=["count", "ngram"], ascending=False)
    df.to_csv(out_path, encoding="utf-8", sep="\t", quoting=csv.QUOTE_NONE)
    print ("Write file {}".format(out_path))


def _gen_ngram_questions(df, aug_dir, file_name):
    ngram_path = "{}/ngram_{}.csv".format(aug_dir, file_name)
    print (df)
    questions = df["question"]
    questions_tokens = []
    for q in questions:
        tokens = [token for token in q.split(" ") if token != ""]
        questions_tokens.append(tokens)

    _write_ngram(questions_tokens, aug_dir, file_name, questions, 1)
    _write_ngram(questions_tokens, aug_dir, file_name, questions, 2)
    _write_ngram(questions_tokens, aug_dir, file_name, questions, 3)
    _write_ngram(questions_tokens, aug_dir, file_name, questions, 4)


def ngram_questions(aug_dir, datasets=None, file_names=None):
    file_paths = []

    if file_names:
        for file_name in file_names:
            file_paths.append("{}\\{}.csv".format(aug_dir, file_name))

    if datasets:
        for dataset in datasets:
            file_paths.append("{}\\entityquestion_{}.csv".format(aug_dir, dataset))
            file_paths.append("{}\\question_{}.csv".format(aug_dir, dataset))

    if not datasets and not file_names:
        file_paths = glob.glob("{}/[!ngram]*.csv".format(aug_dir))

    for file_path in file_paths:
        print (file_path)
        file_name = file_path.split("\\")[-1].replace(".csv", "")
        print (file_name)
        df = pd.read_csv(file_path, encoding="utf-8", sep="\t", quoting=csv.QUOTE_NONE)
        _gen_ngram_questions(df, aug_dir, file_name)


if __name__ == "__main__":

    arg1 = sys.argv[1]

    if arg1 == "summary":
        # summary_tsv_files("qna_data/glue_data/vi", ["train", "dev"], out_dir="qna_data/glue_data/aug")
        # summary_tsv_files("qna_data/glue_data/vi", ["train"], out_dir="qna_data/glue_data/aug")
        # summary_tsv_files("qna_data/glue_data/vi", ["dev"], out_dir="qna_data/glue_data/aug")
        # summary_tsv_files("qna_data/glue_data/vi", ["test"], out_dir="qna_data/glue_data/aug")
        summary_tsv_files("qna_data/glue_data/vi", ["train", "dev"], out_dir="qna_data/glue_data/aug", punc=True)
        summary_tsv_files("qna_data/glue_data/vi", ["train"], out_dir="qna_data/glue_data/aug", punc=True)
        summary_tsv_files("qna_data/glue_data/vi", ["dev"], out_dir="qna_data/glue_data/aug", punc=True)
        summary_tsv_files("qna_data/glue_data/vi", ["test"], out_dir="qna_data/glue_data/aug", punc=True)
    elif arg1 == "ngram_questions":
        ngram_questions("qna_data/glue_data/aug")
        # ngram_questions("qna_data/glue_data/aug", ["train", "dev", "test"])
        # ngram_questions(
        #     "models/bert-base-multilingual-cased-domain",
        #     file_names=["dev_bert-base-multilingual-cased-domain_512_8_3.0_qnli_details_right",
        #                 "dev_bert-base-multilingual-cased-domain_512_8_3.0_qnli_details_wrong"])
