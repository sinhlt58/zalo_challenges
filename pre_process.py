
import re

from nltk.tokenize import word_tokenize, RegexpTokenizer

from utils import read_json_data, write_json_data, get_method_key


class PreProcessor:

    def __init__(self):
        pass

    def pre_process_qna(self, train_file, test_file, method="normal"):
        train_json = read_json_data(train_file)
        test_json = read_json_data(test_file)

        pre_title_key = get_method_key("title", method)
        pre_question_key = get_method_key("question", method)
        pre_text_key = get_method_key("text", method)
        pre_paragraphs_key = get_method_key("paragraphs", method)

        for train_sample in train_json:
            train_sample[pre_title_key] = self.pre_process_text(train_sample["title"])
            train_sample[pre_question_key] = self.pre_process_text(train_sample["question"])
            train_sample[pre_text_key] = self.pre_process_text(train_sample["text"])

        for test_sample in test_json:
            test_sample[pre_title_key] = self.pre_process_text(test_sample["title"])
            test_sample[pre_question_key] = self.pre_process_text(test_sample["question"])

            pre_paragraphs = []
            for p in test_sample["paragraphs"]:
                pre_paragraphs.append({
                    "id": p["id"],
                    "text": self.pre_process_text(p["text"]),
                })
            test_sample[pre_paragraphs_key] = pre_paragraphs

        train_file_name = train_file.split("/")[1]
        test_file_name = test_file.split("/")[1]
        pre_train_file_name = get_method_key(train_file_name, method)
        pre_test_file_name = get_method_key(test_file_name, method)

        write_json_data(train_file.replace(train_file_name, pre_train_file_name), train_json)
        write_json_data(test_file.replace(test_file_name, pre_test_file_name), test_json)

    def pre_process_text(self, text):

        # remove chu thich
        text = re.sub(r'\(([^)]+)\)', "", text)

        # remove punctuation

        # split
        regex_tokenizer = RegexpTokenizer(r'\w+')
        tokens = regex_tokenizer.tokenize(text)

        text = " ".join(tokens)

        return text

if __name__ == "__main__":

    pre_processor = PreProcessor()

    pre_processor.pre_process_qna("qna_data/train.json", "qna_data/test.json")
