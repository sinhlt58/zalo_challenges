import time
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from dataset import Dataset
from models import AttentiveReader

def get_data(num_sample, batch_size, full=False):
    qna_dataset = Dataset.from_json(method="normal")
    qna_dataset.summary()

    if full:
        train_questions, train_paragraphs, train_labels, \
        testset = qna_dataset.get_sample(
            full=True
        )
    else:
        train_questions, train_paragraphs, train_labels, \
        testset = qna_dataset.get_sample(
            num_sample=num_sample,
            debug=False,
        )

    BUFFER_SIZE = len(train_questions)
    BATCH_SIZE = batch_size
    steps_per_epoch = len(train_questions) // BATCH_SIZE

    dataset = tf.data.Dataset.from_tensor_slices(
        (train_questions, train_paragraphs, train_labels)
    ).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    vocab_size = len(qna_dataset.vocab)

    return dataset, steps_per_epoch, vocab_size, qna_dataset, testset

# Create model
def create_model(vocab_size):
    embedding_dim = 100
    q_units = 64
    p_units = 64

    attentive_reader = AttentiveReader(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        q_units=q_units,
        p_units=p_units,
    )

    return attentive_reader

# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#     from_logits=True, reduction='none'
# )

# loss function
def loss_function(pred, real):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=real,
        logits=pred,
    )

    return tf.math.reduce_mean(loss)

# forward and backward step
# @tf.function
def train_step(model, optimizer, questions, paragraphs, labels):

    with tf.GradientTape() as tape:
        pred = model(questions, paragraphs)

        loss = loss_function(pred, labels)

    gradients = tape.gradient(loss, model.trainable_variables)
    # print ("gradients[1]: ", gradients[3])
    # for idx, gradient in enumerate(gradients):
    #     print ("gradient shape: ", gradient)
    #     break
        # print ("gradient {} shape: {}".format(idx, gradient.shape))
        # print ("trainable_variables {} shape: {}".format(idx, model.trainable_variables[idx].shape))
        # print ("trainable_variables shape: ", len(model.trainable_variables))

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

def test(model, questions, paragraphs, labels, mode="dev"):
    logits = model(questions, paragraphs)
    pred_bool = logits >= 0.5

    if mode == "dev":
        true_bool = labels == 1

        # print ("pred_bool: ", pred_bool)
        # print ("true_bool: ", true_bool)
        # print ("(pred_bool == true_bool).sum(): ", (pred_bool == true_bool).numpy().sum())
        # print ("len(logits): ", len(logits))

        accuracy = (pred_bool == true_bool).numpy().sum() / len(logits)
        # print ("accuracy: {}".format(accuracy))
        f1 = f1_score(true_bool, pred_bool, labels=np.unique(pred_bool))

        return accuracy, f1

    elif mode == "eval":
        return pred_bool

# train loop
def run(mode="train", full=False):
    checkpoint_path = "./models/attentive_reader/checkpoint"

    if mode == "check":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-3,
        )

        dataset, steps_per_epoch, vocab_size, qna_dataset, testset = get_data(400, 64, full=False)
        # Forward example to make sure it works
        model = create_model(vocab_size)
        example_questions, example_paragraphs, example_labels = next(iter(dataset))
        sample_output = model(example_questions, example_paragraphs)
        model.summary()
        print ("sample_output shape: ", sample_output.shape)
        loss = train_step(model, optimizer, example_questions, example_paragraphs, example_labels)
        print ("loss: {}".format(loss))

    if mode == "train":
        dataset, steps_per_epoch, vocab_size, qna_dataset, testset = get_data(8000, 64, full)
        model = create_model(vocab_size)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-3,
        )

        EPOCHS = 10

        for epoch in range(EPOCHS):
            start = time.time()

            total_loss = 0

            for (batch, (questions, paragraphs, labels)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = train_step(model, optimizer, questions, paragraphs, labels)
                total_loss += batch_loss

            dev_accuracy = 0
            dev_f1 = 0

            if not full:
                dev_questions, dev_paragraphs, dev_labels = testset[0], testset[1], testset[2]
                dev_accuracy, dev_f1 = test(model, dev_questions, dev_paragraphs, dev_labels)

            # train_accuracy, train_f1 = test(train_questions, train_paragraphs, train_labels)

            train_accuracy = 0
            train_f1 = 0

            print (
                'Epoch {} Loss {:.4f}. Train accuracy {:.4f} f1 {:.4f}.'
                'Dev accuracy {:.4f} f1 {:.4f}. Time taken for 1 epoch {:.4f} sec\n'.format(
                    epoch + 1, total_loss / steps_per_epoch, train_accuracy, train_f1,
                    dev_accuracy, dev_f1,
                    time.time() - start
                )
            )

        print ("Saved model to path: ", checkpoint_path)
        model.save_weights(checkpoint_path)

    if full and mode == "eval":
        qna_dataset = Dataset.from_json(method="normal", mode="test")

        model = create_model(len(qna_dataset.vocab))
        model.load_weights(checkpoint_path)

        test_questions, test_paragraphs = qna_dataset.get_test_data()
        y_preds = []

        batch_size = 50
        for i in range(0, len(test_questions), batch_size):
            batch_test_questions = test_questions[i:i+batch_size]
            batch_test_paragraphs = test_paragraphs[i:i+batch_size]
            y_pred = test(model, batch_test_questions, batch_test_paragraphs, None, "eval")
            y_preds += y_pred.numpy().reshape(len(y_pred),).tolist()

        print ("y_preds length: ", len(y_preds))
        qna_dataset.gen_submit(y_preds)

if __name__ == "__main__":

    mode = sys.argv[1]

    if mode == "check":
        run(mode="check", full=False)

    elif mode == "dev_train":
        run(mode="train", full=False)

    elif mode == "full_train":
        run(mode="train", full=True)

    elif mode == "eval": # generate sumbit file also
        run(mode="eval", full=True)
