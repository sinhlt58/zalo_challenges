# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
# sinh.luutruong added
import scipy
import pandas as pd
# sinh.luutruong end

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer,
                                  DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer)

from transformers import AdamW, WarmupLinearSchedule

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

logger = logging.getLogger(__name__)

# sinh.luutruong added for test set START
from transformers.data.processors.glue import QnliProcessor

class QnliextProcessor(QnliProcessor):
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "test")

    def get_examples_by_file(self, data_dir, file_name):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "{}.tsv".format(file_name))),
            file_name)
# sinh.luutruong added for test set END

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig,
                                                                                RobertaConfig, DistilBertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            class_weight = torch.FloatTensor([1.5776614310645725, 0.7319838056680162]) # sinh added
            class_weight = class_weight.to(args.device)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3],
                      'class_weight': class_weight,}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 and not args.tpu:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.tpu:
                args.xla_model.optimizer_step(optimizer, barrier=True)
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", eval_test=False): # sinh.luutruong added eval_test
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        # sinh.luutruong start
        if eval_test:
            eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=False, eval_test=True)
        else:
            eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        # sinh.luutruong end

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            # sinh.luutruong start
            results.update({
                "logits": preds,
                "probs": scipy.special.softmax(preds, axis=1),
            })
            # sinh.luutruong end
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics("mrpc", preds, out_label_ids) # sinh.luutruong changed eval_task to mrpc to get f1 score
        # sinh.luutruong start
        results.update({
            "preds": preds,
            "true_labels": out_label_ids,
        })
        # sinh.luutruong end
        results.update(result)

        # sinh.luutruong changed
        if not eval_test:
            output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(prefix))
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False, eval_test=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # sinh.luutruong start
    if eval_test:
        processor = QnliextProcessor()
    else:
        processor = QnliextProcessor()
    if eval_test:
        cache_set_name = args.test_file_name
    else:
        cache_set_name = args.dev_file_name if evaluate else args.train_file_name
    # sinh.luutruong end
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        cache_set_name, # sinh.luutruong changed
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        # sinh.luutruong start
        if eval_test:
            examples = processor.get_examples_by_file(args.data_dir, args.test_file_name)
        else:
            if evaluate:
                examples = processor.get_examples_by_file(args.data_dir, args.dev_file_name)
            else:
                examples = processor.get_examples_by_file(args.data_dir, args.train_file_name)
        # sinh.luutruong end
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

# sinh.luutruong start
def gen_submit(checkpoint_dir, logits, probs, preds, test_prefix, args, pids_file_name="test_pids"):
    import pandas as pd

    n = probs.shape[0]
    submit_dict = {
        "test_id": [],
        "answer": [],
    }
    test_details_dict = {
        "id": [],
        "pred": [],
        "0_logit": [],
        "1_logit": [],
        "0_prob": [],
        "1_prob": [],
    }

    file_pids = os.path.join(args.data_dir, "{}.txt".format(pids_file_name))
    file_submit = os.path.join(checkpoint_dir, "{}_submit.csv".format(test_prefix))
    file_details = os.path.join(checkpoint_dir, "{}_details.csv".format(test_prefix))

    with open(file_pids, "r") as f:
        lines = f.read().split('\n')

    for i in range(n):
        if not preds[i]: # ATTENTION: 0 means entailment, 1 otherwise
            parts = lines[i].split('@')
            submit_dict["test_id"].append(parts[0])
            submit_dict["answer"].append(parts[1])

        test_details_dict["id"].append(lines[i])
        test_details_dict["pred"].append(preds[i])
        test_details_dict["0_logit"].append(logits[i][0])
        test_details_dict["1_logit"].append(logits[i][1])
        test_details_dict["0_prob"].append(probs[i][0])
        test_details_dict["1_prob"].append(probs[i][1])

    pd.DataFrame(submit_dict).to_csv(file_submit, index=False, columns=["test_id", "answer"])
    pd.DataFrame(test_details_dict).to_csv(file_details, index=False, header=True)

def gen_dev_details(
    checkpoint_dir,
    logits,
    probs,
    preds,
    true_labels,
    test_prefix,
    args,
    pids_file_name="dev_pids"
):
    from sklearn.metrics import classification_report, confusion_matrix
    import io
    import json

    n = probs.shape[0]
    test_details_dict = {
        "id": [],
        "pred": [],
        "true": [],
        "0_prob": [],
        "1_prob": [],
        "correct": [],
        "0_logit": [],
        "1_logit": [],
        "pred_str": [],
        "true_str": [],
    }

    file_pids = os.path.join(args.data_dir, "{}.txt".format(pids_file_name))
    file_details = os.path.join(checkpoint_dir, "{}_details.csv".format(test_prefix))
    file_report = os.path.join(checkpoint_dir, "{}_report.json".format(test_prefix))
    file_confusion = os.path.join(checkpoint_dir, "{}_confusion.csv".format(test_prefix))

    with open(file_pids, "r") as f:
        lines = f.read().split('\n')

    for i in range(n):
        test_details_dict["id"].append(lines[i].split('@')[0])
        test_details_dict["pred"].append(preds[i])
        test_details_dict["true"].append(true_labels[i])
        test_details_dict["correct"].append(1 if preds[i] == true_labels[i] else 0)
        test_details_dict["0_prob"].append(probs[i][0])
        test_details_dict["1_prob"].append(probs[i][1])
        test_details_dict["0_logit"].append(logits[i][0])
        test_details_dict["1_logit"].append(logits[i][1])
        test_details_dict["pred_str"].append("no_answer" if preds[i] else "has_answer")
        test_details_dict["true_str"].append("no_answer" if true_labels[i] else "has_answer")

    # confusion matrix
    df_confusion = pd.crosstab(true_labels, preds, rownames=['Actual'], colnames=['Predicted'], margins=True)
    # classification_report
    target_names = ["has_answer", "no_answer"]
    report = classification_report(true_labels, preds, target_names=target_names, output_dict=True)

    print (report)
    print (df_confusion)

    pd.DataFrame(test_details_dict).to_csv(file_details, index=False, header=True)
    df_confusion.to_csv(file_confusion, index=True, header=True)
    with io.open(file_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    print ("Write file {}".format(file_details))
    print ("Write file {}".format(file_confusion))
    print ("Write file {}".format(file_report))

# sinh.luutruong end

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    # sinh.luutruong added for test set START
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to evaluate the test set.")
    parser.add_argument("--do_private", action='store_true',
                        help="Whether to evaluate the private set.")
    parser.add_argument("--train_file_name", default="train", type=str,
                        help="The train file name.")
    parser.add_argument("--dev_file_name", default="dev", type=str,
                        help="The dev evaluation file name.")
    parser.add_argument("--dev_pids_file_name", default="dev_pids", type=str,
                        help="The pids file name.")
    parser.add_argument("--test_file_name", default="test", type=str,
                        help="The test file name.")
    parser.add_argument("--test_pids_file_name", default="test_pids", type=str,
                        help="The pids file name.")
    parser.add_argument("--from_tf", action="store_true",
                        help="Whether from tf model")
    # sinh.luutruong added for test set END
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--tpu', action='store_true',
                        help="Whether to run on the TPU defined in the environment variables")
    parser.add_argument('--tpu_ip_address', type=str, default='',
                        help="TPU IP address if none are set in the environment variables")
    parser.add_argument('--tpu_name', type=str, default='',
                        help="TPU name if none are set in the environment variables")
    parser.add_argument('--xrt_tpu_config', type=str, default='',
                        help="XRT TPU config if none are set in the environment variables")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    if args.tpu:
        if args.tpu_ip_address:
            os.environ["TPU_IP_ADDRESS"] = args.tpu_ip_address
        if args.tpu_name:
            os.environ["TPU_NAME"] = args.tpu_name
        if args.xrt_tpu_config:
            os.environ["XRT_TPU_CONFIG"] = args.xrt_tpu_config

        assert "TPU_IP_ADDRESS" in os.environ
        assert "TPU_NAME" in os.environ
        assert "XRT_TPU_CONFIG" in os.environ

        import torch_xla
        import torch_xla.core.xla_model as xm
        args.device = xm.xla_device()
        args.xla_model = xm

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # sinh.luutruong changed
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if args.do_train:
        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)

        logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and not args.tpu:
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)


    # Evaluation
    results = {}
    if (args.do_eval or args.do_test) and args.local_rank in [-1, 0]: # sinh.luutruong add do_test
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint, from_tf=args.from_tf)
            model.to(args.device)
            # sinh.luutruong start
            if args.do_eval:
                result = evaluate(args, model, tokenizer, prefix=prefix, eval_test=False) # sinh.luutruong added eval_test

                preds = result["preds"]
                logits = result["logits"]
                probs = result["probs"]
                true_labels = result["true_labels"]
                dev_prefix = '{}_{}_{}_{}_{}_{}'.format(
                                    args.dev_file_name,
                                    list(filter(None, args.model_name_or_path.split('/'))).pop(),
                                    str(args.max_seq_length),
                                    str(args.per_gpu_train_batch_size),
                                    str(args.num_train_epochs),
                                    str(args.task_name))
                print ("dev_prefix: ", dev_prefix)
                gen_dev_details(checkpoint, logits, probs, preds, true_labels, dev_prefix, args, args.dev_pids_file_name)

                result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
                results.update(result)

            test_loop = []
            if args.do_test:
                test_loop.append("test")
            if args.do_private:
                test_loop.append("private")

            for test_type in test_loop:
                if test_type == "private":
                    args.test_file_name = "private"
                    args.test_pids_file_name = "private_pids"

                result = evaluate(args, model, tokenizer, prefix="{}_test".format(prefix), eval_test=True)
                preds = result["preds"]
                logits = result["logits"]
                probs = result["probs"]
                test_prefix = '{}_{}_{}_{}_{}_{}'.format(
                                    args.test_file_name,
                                    list(filter(None, args.model_name_or_path.split('/'))).pop(),
                                    str(args.max_seq_length),
                                    str(args.per_gpu_train_batch_size),
                                    str(args.num_train_epochs),
                                    str(args.task_name))
                print ("test_prefix: ", test_prefix)
                gen_submit(checkpoint, logits, probs, preds, test_prefix, args, args.test_pids_file_name)
            # sinh.luutruong end

    return results


if __name__ == "__main__":
    main()
