import sys
import time
from collections import OrderedDict
import requests
import urllib
import traceback
import os
import glob
import re

from underthesea import sent_tokenize

from utils import read_json_data, get_file_names_in_dir, untokenize

def get_titles():
    dataset_types = ["train", "test"]
    titles_file = "qna_data/wiki_data/titles.txt"

    with open(titles_file, "r", encoding="utf-8") as f:
        titles = f.read().split("\n")[:-1]
        print ("Number of titles before: ", len(titles))

    for dataset_type in dataset_types:
        json_samples = read_json_data("qna_data/vi_{}.json".format(dataset_type))

        for json_sample in json_samples:
            title = json_sample["title"].strip().replace(" ", " ")
            if title and title not in titles:
                titles.append(title)

    print ("Number of titles after: ", len(titles))
    with open(titles_file, "w", encoding="utf-8") as f:
        for title in titles:
            f.write("{}\n".format(title))

def _search_article(untokenized_title):
    api_url = "https://vi.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "srlimit": 10,
        "gsrsearch": untokenized_title,
        "prop": "info",
        "format": "json",
    }
    encoded_parms = urllib.parse.urlencode(params)
    res = requests.get("{}?{}".format(api_url, encoded_parms)).json()
    pages = res["query"]["pages"]
    found_titles = []

    for k, page_info in pages.items():
        found_titles.append(page_info["title"])

    return found_titles

def _write_article(title, text, wiki_dir="articles_data", overwrite=False):
    write_file = "qna_data/wiki_data/{}/{}.txt".format(wiki_dir, title)
    if overwrite or (not overwrite and not os.path.exists(write_file)):
        with open(write_file, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print ("File {} already exists so we dont write it".format(write_file))

def _download_wiki_article(title, download_title):
    api_url = "https://vi.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": download_title,
        "prop": "extracts",
        "explaintext": "",
    }
    encoded_parms = urllib.parse.urlencode(params)
    res = requests.get("{}?{}".format(api_url, encoded_parms)).json()
    pages = res["query"]["pages"]
    for page_id, page_data in pages.items():
        if title == page_data["title"]:
            text = page_data["extract"]

            # we raise exception for empty page so we find related pages later
            if not text:
                print ("Empty text for title {}".format(title))
                raise Exception("Empty text")
            else:
                _write_article(title, text)

def _get_untokenized_title(title):
    untokenized_title = untokenize(title.split())
    if untokenized_title[-2:] == " )":
        untokenized_title = untokenized_title.replace(" )", ")")
    return untokenized_title

def download_articles():
    titles_file = "qna_data/wiki_data/titles.txt"
    with open(titles_file, "r", encoding="utf-8") as f:
        titles = f.read().split("\n")[:-1]

    downloaded_title_names = get_file_names_in_dir("qna_data/wiki_data/articles_data")
    for title in titles:
        if title not in downloaded_title_names:
            untokenized_title = _get_untokenized_title(title)
            untokenized_title = untokenized_title.replace(" ", "_")

            try:
                _download_wiki_article(title, untokenized_title)
                time.sleep(0.3)
                print ("Downloaded and saved {} - {} ".format(title, download_title))
            except Exception as e:
                print ("Error while downloading so search new article for {} - {}".format(title, untokenized_title))

                try:
                    # we download found titles and save empty error article
                    found_titles = _search_article(untokenized_title)

                    for found_title in found_titles:
                        if found_title not in downloaded_title_names:
                            _download_wiki_article(found_title, found_title)
                            time.sleep(0.3)
                            print ("Download and saved found title {}".format(found_title))

                    # write empty article
                    _write_article(title, "")

                except Exception as e:
                    print ("Error while download title {} so skip".format(found_title))
                    traceback.print_exc()

def _is_in_exclude_headings(heading_text):
    patterns = [
        r"=== Xem thêm ===",
        r"=== Chú thích ===",
        r"=== Ghi chú ===",
        r"=== Tham khảo ===",
        r"=== Sách ===",
        r"=== Online ===",
        r"=== Liên kết ngoài ===",
        r"=== ===",
    ]
    for pattern in patterns:
        if re.match(pattern, heading_text):
            return True
    return False

def _is_valid_paragraph(p):
    return p and len(p) > 50

def _preprocess_paragraph(p):
     # we don't include meta sections or too short section
    if _is_in_exclude_headings(p) or \
        not _is_valid_paragraph(p):
        return ""

    is_heading = re.match(r"=== .+ ===", p)
    # skip the heading paragraph
    if is_heading:
        return ""

    tokens = p.split()
    p = " ".join(tokens)

    return p.strip()

def _is_valid_sent(sent):
    return len(sent) > 10

def _get_paragraph_bert_sentences(p):
    res = ""

    for sent in sent_tokenize(p):
        if _is_valid_sent(sent):
            res += sent + "\n"

    return res.strip()

def _preprocess_wiki_section(section_text):
    paragraphs = section_text.split("\n")
    res_text = ""
    for p in paragraphs:
        p = _preprocess_paragraph(p)
        p = _get_paragraph_bert_sentences(p)

        if p:
            res_text += p + "\n\n"

    return res_text

def _preprocess_raw_wiki(file_name, articles_dir, pre_articles_dir):
    input_file = "{}/{}.txt".format(articles_dir, file_name)
    output_file = "{}/{}.txt".format(pre_articles_dir, file_name)

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    pre_text = ""
    if text:
        sections = text.split("\n\n\n")
        for idx, section in enumerate(sections):
            pre_section = _preprocess_wiki_section(section)
            pre_text += pre_section

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(pre_text)


def to_bert(article_prefix_dir=""):
    articles_dir = "qna_data/wiki_data/{}/articles_data".format(article_prefix_dir)
    pre_articles_dir = "qna_data/wiki_data/{}/pre_articles".format(article_prefix_dir)

    for file_name in get_file_names_in_dir(articles_dir):
        _preprocess_raw_wiki(file_name, articles_dir, pre_articles_dir)

def clean_pre_articles():
    # delete duplicate and empty file
    pre_articles_dir = "qna_data/wiki_data/pre_articles"

    valid_titles = []
    need_to_delete_titles = []

    for file_name in get_file_names_in_dir(pre_articles_dir):
        file_path = "qna_data/wiki_data/pre_articles/{}.txt".format(file_name)
        untokenized_title = _get_untokenized_title(file_name)
        file_size = os.stat(file_path).st_size

        if untokenized_title in valid_titles or file_size == 0:
            os.remove(file_path)
        else:
            valid_titles.append(untokenized_title)

    print ("deleted {} files".format(len(need_to_delete_titles)))

def concat(article_prefix_dir=""):
    pre_articles_dir = "qna_data/wiki_data/{}/pre_articles".format(article_prefix_dir)
    domain_file = "qna_data/wiki_data/{}/domain.txt".format(article_prefix_dir)

    with open(domain_file, "w", encoding="utf-8") as outf:
        for file_path in glob.glob("{}/*.txt".format(pre_articles_dir)):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    outf.write(line)

def count():
    domain_file = "qna_data/wiki_data/domain.txt"

    num_line = 0
    with open(domain_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                num_line += 1

    batch_size = 32
    num_epochs = 1
    warnup_ratio = 0.1
    max_seq_length = 512
    mask_prob = 0.15

    num_train_steps = int(num_line / batch_size * num_epochs)
    num_warmup_steps = int(num_train_steps * warnup_ratio)
    max_predictions_per_seq = max_seq_length * mask_prob

    print ("num_line: ", num_line)
    print ("batch_size: ", batch_size)
    print ("num_epochs: ", num_epochs)
    print ("warnup_ratio: ", warnup_ratio)
    print ("num_train_steps: ", num_train_steps)
    print ("num_warmup_steps: ", num_warmup_steps)
    print ("max_seq_length: ", max_seq_length)
    print ("mask_prob: ", mask_prob)
    print ("max_predictions_per_seq: ", max_predictions_per_seq)

if __name__ == "__main__":
    arg1 = sys.argv[1]

    if arg1 == "get_titles":
        get_titles()
    elif arg1 == "download":
        download_articles()
    elif arg1 == "to_bert":
        to_bert()
    elif arg1 == "clean_pre":
        clean_pre_articles()
    elif arg1 == "concat":
        concat()
    elif arg1 == "count":
        count()
    else:
        print ("No task: {}".format(arg1))
