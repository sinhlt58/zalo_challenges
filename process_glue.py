import sys
import glob
import csv
import collections
import pandas as pd

import tensorflow as tf
from transformers import (
    BertTokenizer,
    XLMTokenizer,
    glue_convert_examples_to_features,
)
from utils import write_json_data, read_json_data, find_dict_data, process_split_text


def _int64_feature(values):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

def serialize_example(features_dict, label):
    feature = collections.OrderedDict()
    feature["input_ids"] = _int64_feature(features_dict["input_ids"].numpy().tolist())
    feature["attention_mask"] = _int64_feature(features_dict["attention_mask"].numpy().tolist())
    feature["token_type_ids"] = _int64_feature(features_dict["token_type_ids"].numpy().tolist())
    feature["label"] = _int64_feature([int(label.numpy())])

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example.SerializeToString()

TOKENIZER_CLASSES = {
    "bert": BertTokenizer,
    "xlm": XLMTokenizer,
}

def _to_tf_record(tsv_dir, model_type, model_name, max_seq_len, include=None):
    # bert_model = "bert-base-multilingual-cased"
    # max_seq_len = 512

    tokenizer = TOKENIZER_CLASSES[model_type].from_pretrained(model_name)

    for tsv_path in glob.glob("{}/*.tsv".format(tsv_dir)):
        file_name = tsv_path.split("\\")[-1].replace(".tsv", "")
        if include and file_name not in include:
            print ("skip file {}".format(file_name))
            continue

        df = pd.read_csv(tsv_path, encoding="utf-8", quoting=csv.QUOTE_NONE, sep="\t")
        df["idx"] = df["index"]
        tf_dataset = tf.data.Dataset.from_tensor_slices(dict(df))

        features_dataset = glue_convert_examples_to_features(tf_dataset, tokenizer, max_seq_len, 'qnli')
        # print (features_dataset)
        # for features, label in features_dataset:
        #     print (features)
        #     print (label)

        def gen():
            for features_dict, label in features_dataset:
                yield serialize_example(features_dict, label)

        serialized_features_dataset = tf.data.Dataset.from_generator(
            gen, output_types=tf.string, output_shapes=())

        num_examples = df.shape[0]
        file_name_id = "{}@{}@{}@{}@{}".format(
            file_name, model_type, model_name, max_seq_len, num_examples
        )
        record_path = "{}/{}.tfrecord".format(
            tsv_dir, file_name_id
        )
        writer = tf.data.experimental.TFRecordWriter(record_path)
        writer.write(serialized_features_dataset)
        print ("Write file {}".format(record_path))
        # Write meta data
        # meta_dict = {
        #     "model_type": model_type,
        #     "model_name": model_name,
        #     "max_seq_len": max_seq_len,
        #     "num_examples": num_examples,
        # }
        # meta_path = "{}/{}.json".format(tsv_dir, file_name_id)


def _test_read_tf_record(max_seq_len=512):
    name_to_features = {
      "input_ids": tf.io.FixedLenFeature([max_seq_len], tf.int64),
      "attention_mask": tf.io.FixedLenFeature([max_seq_len], tf.int64),
      "token_type_ids": tf.io.FixedLenFeature([max_seq_len], tf.int64),
      "label": tf.io.FixedLenFeature([], tf.int64),
    }

    def _decode_record(example_proto):
        """Decodes a record to a TensorFlow example."""
        example_dict = tf.io.parse_single_example(example_proto, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        features_dict = {}
        for name in list(example_dict.keys()):
            t = example_dict[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example_dict[name] = t
            if name != "label":
                features_dict[name] = t

        label = example_dict["label"]

        return (features_dict, label)

    record_path = "qna_data/glue_data/vi/dev.tfrecord"
    filenames = [record_path]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    parsed_dataset = raw_dataset.map(_decode_record)
    parsed_dataset = parsed_dataset.batch(4)

    for batch_features, batch_label in parsed_dataset.take(2):
        print (batch_features)

def _get_sort_dist(count_dict):
    data_dict = {
        "length": [],
        "count": [],
    }

    for k, v in count_dict.items():
        data_dict["length"].append(k)
        data_dict["count"].append(v)

    df = pd.DataFrame(data_dict)
    df = df.sort_values(by="length", ascending=False)
    return df

def _get_tsv_summary(tsv_file):
    df = pd.read_csv(tsv_file, encoding="utf-8", quoting=csv.QUOTE_NONE, sep="\t")
    res_dict = None
    columns = list(df.columns)

    if "label" in columns:
        total = int(df.shape[0])
        num_has_answer = int((df["label"] == "entailment").sum())
        num_no_answer = total - num_has_answer
        res_dict = {
            "total": total,
            "num_has_answer": num_has_answer,
            "num_no_answer": num_no_answer,
            "percentage_has_answer": num_has_answer / total,
            "percentage_no_answer": num_no_answer / total,
            "has_answer_ratio": num_has_answer / num_no_answer
        }

    question_len_dist = {}
    sentence_len_dist = {}
    outlier_dict = {
        "question": [],
        "sentence": [],
        "question_len": [],
        "sentence_len": [],
    }
    q_len_df = None
    s_len_df = None
    outlier_df = None

    if "question" in columns and "sentence" in columns:
        for i in range(0, df.shape[0]):
            len_q = len(df["question"][i].split())
            len_s = len(df["sentence"][i].split())

            if len_s <= 10 or len_q <= 5 or len_s > 250:
                outlier_dict["question"].append(df["question"][i])
                outlier_dict["sentence"].append(df["sentence"][i])
                outlier_dict["question_len"].append(len_q)
                outlier_dict["sentence_len"].append(len_s)

            if len_q not in question_len_dist:
                question_len_dist[len_q] = 0
            question_len_dist[len_q] += 1
            if len_s not in sentence_len_dist:
                sentence_len_dist[len_s] = 0
            sentence_len_dist[len_s] += 1

        q_len_df = _get_sort_dist(question_len_dist)
        s_len_df = _get_sort_dist(sentence_len_dist)
        outlier_df = pd.DataFrame(outlier_dict).sort_values(by=["sentence_len", "question_len"])

    return res_dict, q_len_df, s_len_df, outlier_df


def _summary(tsv_dir):
    summary_dict = {}
    for tsv_path in glob.glob("{}/*.tsv".format(tsv_dir)):
        file_name = tsv_path.split("\\")[-1].replace(".tsv", "")
        res_dict, q_len_df, s_len_df, outlier_df = _get_tsv_summary(tsv_path)

        if res_dict:
            summary_dict[file_name] = res_dict
        if q_len_df is not None and s_len_df is not None:
            q_len_df.to_csv("{}/{}_question_length_dist.csv".format(tsv_dir, file_name), header=True, index=False)
            s_len_df.to_csv("{}/{}_sentence_length_dist.csv".format(tsv_dir, file_name), header=True, index=False)
        if outlier_df is not None:
            outlier_df.to_csv(
                "{}/{}_outlier.csv".format(tsv_dir, file_name),
                header=False, encoding="utf-8", sep="\t", quoting=csv.QUOTE_NONE)

    summary_path = "{}/summary.json".format(tsv_dir)
    write_json_data(summary_path, summary_dict)
    print ("Write file {}".format(summary_path))

def _write_dev_details(df, file_path, data_dict):
    questions = []
    texts = []

    for _id in df["id"]:
        sample = find_dict_data(_id, data_dict)
        questions.append(process_split_text(sample["question"]))
        texts.append(process_split_text(sample["text"]))
    df["question"] = questions
    df["text"] = texts

    df = df.sort_values(by=["true", "0_prob", "text"])
    df.to_csv(file_path, index=False, encoding="utf-8", sep="\t", quoting=csv.QUOTE_NONE, float_format='%.4f')
    print ("Write file {} shape {}".format(file_path, df.shape))

def _gen_wrong_dev(model_dir, details_file_name, json_data_path):
    details_path = "{}/{}.csv".format(model_dir, details_file_name)
    details_df = pd.read_csv(details_path)
    data_dict = read_json_data(json_data_path)

    wrong_path = "{}/{}_wrong.csv".format(model_dir, details_file_name)
    wrong_df = details_df.loc[details_df["correct"] == 0]
    _write_dev_details(wrong_df, wrong_path, data_dict)

    right_path = "{}/{}_right.csv".format(model_dir, details_file_name)
    right_df = details_df.loc[details_df["correct"] == 1]
    _write_dev_details(right_df, right_path, data_dict)


if __name__ == "__main__":
    arg1 = sys.argv[1]

    if arg1 == "to_tf_record":
        # _to_tf_record("qna_data/glue_data/vi", "bert", "bert-base-multilingual-cased", 512)
        _to_tf_record("qna_data/glue_data/vi", "bert", "bert-base-multilingual-cased", 512, include=["train", "dev", "test"])
        # _to_tf_record("qna_data/glue_data/vi", "xlm", "xlm-mlm-17-1280", 256)
        # _to_tf_record("qna_data/glue_data/vi", "xlm", "xlm-mlm-17-1280", 128)
    elif arg1 == "test_read_tf_record":
        _test_read_tf_record()
    elif arg1 == "summary":
        _summary("qna_data/glue_data/vi")
    elif arg1 == "gen_wrong_dev":
        _gen_wrong_dev(
            "models/bert-base-multilingual-cased-domain",
            "dev_bert-base-multilingual-cased-domain_512_8_3.0_qnli_details",
            "qna_data/train.json"
        )
