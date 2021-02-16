import os
import random
import logging

import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel, AutoConfig


def get_label(args):
    return [0, 1]


def load_tokenizer(args):
    return AutoTokenizer.from_pretrained(args.model_name_or_path)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_score(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_score(preds, labels):
    return {
        "acc": simple_accuracy(preds, labels),
    }
