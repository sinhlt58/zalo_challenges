import os
import csv
import ast
import tensorflow as tf
import tensorflow_datasets
from transformers import (
    BertTokenizer,
    TFBertForSequenceClassification,
    glue_convert_examples_to_features,
    BertForSequenceClassification,
)

def _load_glue_dataset(file_path):
    dict_data = {
        "idx": [],
        "label": [],
        "question": [],
        "sentence": [],
    }
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader, 0) # skip header

        for row in reader:
            dict_data["idx"].append(int(row[0]))
            dict_data["question"].append(row[1])
            dict_data["sentence"].append(row[2])
            dict_data["label"].append(row[3])

    tf_dataset = tf.data.Dataset.from_tensor_slices(dict_data)
    return tf_dataset, len(dict_data["label"])

def load_glue_data(folder_path, task="qnli"):
    train_set, train_len = _load_glue_dataset("{}/{}/train.tsv".format(folder_path, task))
    dev_set, dev_len = _load_glue_dataset("{}/{}/dev.tsv".format(folder_path, task))

    return (train_set, train_len), (dev_set, dev_len)

# script parameters
BATCH_SIZE = 32
EVAL_BATCH_SIZE = BATCH_SIZE * 2
USE_XLA = False
USE_AMP = False

# tf.config.optimizer.set_jit(USE_XLA)
# tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})

# if USE_AMP:
#     # loss scaling is currently required when using mixed precision
#     opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')

resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)

# Load dataset via TensorFlow Datasets
data_folder = "/content/drive/My Drive/AI/data/glue_data"
(train_raw, train_examples), (dev_raw, valid_examples) = load_glue_data(data_folder, "qnli")


# Load tokenizer and model from pretrained model/vocabulary
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

train_steps = train_examples//BATCH_SIZE
valid_steps = valid_examples//EVAL_BATCH_SIZE

# Prepare dataset for GLUE as a tf.data.Dataset instance
train_dataset = glue_convert_examples_to_features(train_raw, tokenizer, 300, 'qnli')
valid_dataset = glue_convert_examples_to_features(dev_raw, tokenizer, 300, 'qnli')
train_dataset = train_dataset.shuffle(300).batch(BATCH_SIZE).repeat(-1)
valid_dataset = valid_dataset.batch(EVAL_BATCH_SIZE)

with strategy.scope():
    # Train and evaluate using tf.keras.Model.fit()

    # distributed_dataset = iter(strategy.experimental_distribute_dataset(dataset))
    # Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
    opt = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')
    model.compile(optimizer=opt, loss=loss, metrics=[metric])

    history = model.fit(train_dataset, epochs=2, steps_per_epoch=train_steps,
                        validation_data=valid_dataset, validation_steps=valid_steps)

# Save TF2 model
os.makedirs('./save/', exist_ok=True)
model.save_pretrained('./save/')

# Load the TensorFlow model in PyTorch for inspection
# pytorch_model = BertForSequenceClassification.from_pretrained('./save/', from_tf=True)

# # Quickly test a few predictions - MRPC is a paraphrasing task, let's see if our model learned the task
# sentence_0 = 'This research was consistent with his findings.'
# sentence_1 = 'His findings were compatible with this research.'
# sentence_2 = 'His findings were not compatible with this research.'
# inputs_1 = tokenizer.encode_plus(sentence_0, sentence_1, add_special_tokens=True, return_tensors='pt')
# inputs_2 = tokenizer.encode_plus(sentence_0, sentence_2, add_special_tokens=True, return_tensors='pt')

# pred_1 = pytorch_model(**inputs_1)[0].argmax().item()
# pred_2 = pytorch_model(**inputs_2)[0].argmax().item()
# print('sentence_1 is', 'a paraphrase' if pred_1 else 'not a paraphrase', 'of sentence_0')
# print('sentence_2 is', 'a paraphrase' if pred_2 else 'not a paraphrase', 'of sentence_0')
