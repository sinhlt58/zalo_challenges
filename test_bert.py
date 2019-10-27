

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from bert_official.tokenization import FullTokenizer
from bert import BertModelLayer
from bert import params_from_pretrained_ckpt


MODEL_PATH = "D:/works/zalo_challenges/models/bert/uncased_L-24_H-1024_A-16"
VOCAB_FILE = '{}/vocab.txt'.format(MODEL_PATH)

tokenizer = FullTokenizer(
    vocab_file=VOCAB_FILE,
    do_lower_case=True
)

examples = [
    "Open source operating system is most used today",
    "Android source code is licensed under free and open source software licenses. Google brings most of the code (including network and phone layers) under the Apache License Version 2.0, and the rest, changes to the Linux kernel, under the GNU General Public License version 2. The Open Handset Alliance made changes to the Linux kernel, with the source code publicly available. Google develops by itself, and the source code is only released when a new version is released.Usually, Google works with a hardware manufacturer to provide a flagship device (Google Nexus series) with the version. latest Android version, then release the source code after this device is sold.",
    "What year was the first World Cup?",
    "At the international level, the 1954 World Cup was the first major television broadcast. Right from the beginning, the relationship between television and football had many conflicts. Matt Busby, coach of Manchester United declared in 1957 Football players must be paid for their value.",
    "The biggest international tournament of world football is the World Cup. The World Cup was first held by FIFA in 1930 and has now become the most watched sports competition on the planet, surpassing the Olympics, such as the 2006 World Cup final held in Germany. attracted 26.29 billion viewers watching television in which the final alone has attracted 715.1 million well-off people around the world."
]

list_tokens = []

for example in examples:
    tokens = tokenizer.tokenize(example)
    list_tokens.append(tokens)
    # print ("tokens: ", tokens)
    # token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print ("token_ids: ", token_ids)
    # list_token_ids.append(token_ids)

list_token_ids = []
padd_list_tokens = pad_sequences(list_tokens, dtype=object, value='[PAD]', padding='post')

for padd_tokens in padd_list_tokens:
    token_ids = tokenizer.convert_tokens_to_ids(padd_tokens)
    list_token_ids.append(token_ids)
    print ("padd_tokens: ", padd_tokens)
    print ("token_ids: ", token_ids)

class TestModel(tf.keras.Model):

    def __init__(self, model_path):
        super().__init__()

        bert_params = params_from_pretrained_ckpt(model_path)
        self.bert_layer = BertModelLayer.from_params(bert_params)
        self.bert_layer.trainable = False

    def call(self, x):

        return self.bert_layer(x)

test_model = TestModel(MODEL_PATH)
example_x = tf.convert_to_tensor(list_token_ids)
example_out = test_model(example_x)

test_model.summary()

print ("example_out: ", example_out)
