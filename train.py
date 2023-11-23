from __future__ import absolute_import, division, print_function

import argparse
import random
import logging
import sys
import pickle
import numpy as np
from pathlib import Path

import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from bert import MAG_BertForSequenceClassification

from argparse_utils import seed
from data_loader import create_dataset
from global_configs import *

parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--dev_batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=40)
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=seed, default='random')
parser.add_argument("--save_path", type=str, default="./checkpoint/model.pt")

pwd = os.path.abspath(__file__)
logging.basicConfig(
            filename=os.path.join(Path(pwd).parent, "results.log"),
            format='%(asctime)s %(message)s',
            datefmt='%m/%d/%Y %p %I:%M:%S',
            level=logging.INFO
        )
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("trainLogger")
logger.addHandler(logging.StreamHandler(sys.stdout))

args = parser.parse_args()


def set_random_seed(seed: int):
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob


def set_up_data_loader():
    train_file = open("./train.pkl", "rb")
    train_d = pickle.load(train_file)

    dev_file = open("./val.pkl", "rb")
    dev_d = pickle.load(dev_file)

    train_data = train_d["train"]
    dev_data = dev_d["val"]

    train_dataset = create_dataset(train_data)
    dev_dataset = create_dataset(dev_data)

    num_train_optimization_steps = (
            int(
                len(train_dataset) / args.train_batch_size /
                args.gradient_accumulation_step
            )
            * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=True
    )

    return (
        train_dataloader,
        dev_dataloader,
        num_train_optimization_steps,
    )


def prep_for_training(num_train_optimization_steps: int):
    multimodal_config = MultimodalConfig(
        beta_shift=args.beta_shift, dropout_prob=args.dropout_prob
    )
    model = MAG_BertForSequenceClassification.from_pretrained(
        'kykim/bert-kor-base', multimodal_config=multimodal_config, num_labels=1,
    )
    model.to(DEVICE)

    # Prepare optimizer
    model_param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model_param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in model_param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_train_optimization_steps,
        num_training_steps=args.warmup_proportion * num_train_optimization_steps,
    )
    return model, optimizer, scheduler


def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        acoustic = batch['audio'].to(DEVICE)
        visual = batch['video'].to(DEVICE)
        input_ids = batch['input_ids'].to(DEVICE)
        segment_ids = batch['segment_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        label_ids = batch['label'].to(DEVICE)
        outputs = model(input_ids, visual, acoustic.float(), attention_mask, segment_ids)
        logits = outputs[0]
        loss_fct = nn.MSELoss()
        loss = loss_fct(logits.view(-1), label_ids.view(-1))

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        loss.backward()

        tr_loss += loss.item()
        nb_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return tr_loss / nb_tr_steps


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            acoustic = batch['audio'].to(DEVICE)
            visual = batch['video'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            segment_ids = batch['segment_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            label_ids = batch['label'].to(DEVICE)

            outputs = model(input_ids, visual, acoustic.float(), attention_mask, segment_ids)
            logits = outputs[0]

            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            nb_dev_steps += 1

    return dev_loss / nb_dev_steps


def train(
    model,
    train_dataloader,
    validation_dataloader,
    optimizer,
    scheduler,
):

    best_valid = 1e8
    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        valid_loss = eval_epoch(model, validation_dataloader)
        logger.info("epoch:{}, train_loss:{}, valid_loss:{}".format(epoch_i, train_loss, valid_loss))
        if valid_loss < best_valid:
            torch.save({
                'epoch': epoch_i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, args.save_path)
            print("*****************MODEL SAVED*****************")
            best_valid = valid_loss


def main():
    set_random_seed(args.seed)
    logger.info("seed:{}".format(args.seed))

    (
        train_data_loader,
        dev_data_loader,
        num_train_optimization_steps,
    ) = set_up_data_loader()

    model, optimizer, scheduler = prep_for_training(
        num_train_optimization_steps)

    train(
        model,
        train_data_loader,
        dev_data_loader,
        optimizer,
        scheduler,
    )


if __name__ == "__main__":
    main()
