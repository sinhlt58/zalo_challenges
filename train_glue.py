import tensorflow as tf
import tensorflow_datasets
from transformers import *

# Load dataset, tokenizer, model from pretrained model/vocabulary
tokenizer = RobertaTokenizer.from_pretrained('roberta-large', cache_dir="models/roberta-large")
model = TFRobertaForSequenceClassification.from_pretrained('roberta-large')
data = tensorflow_datasets.load('glue/qnli', download=True, data_dir="glue_data")

# Prepare dataset for GLUE as a tf.data.Dataset instance
train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=300, task='qnli')
valid_dataset = glue_convert_examples_to_features(data['validation'], tokenizer, max_length=300, task='qnli')
train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
valid_dataset = valid_dataset.batch(64)

# Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# # Train and evaluate using tf.keras.Model.fit()
history = model.fit(train_dataset, epochs=2, steps_per_epoch=115,
                    validation_data=valid_dataset, validation_steps=7)

# # Load the TensorFlow model in PyTorch for inspection
# model.save_pretrained('./save/')
# pytorch_model = BertForSequenceClassification.from_pretrained('./save/', from_tf=True)
