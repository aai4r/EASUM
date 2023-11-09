from __future__ import absolute_import, division, print_function

import argparse
import os
import random
import pickle
import logging
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from transformers import ConvBertTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from convbert import MAG_ConvBertForSequenceClassification

from argparse_utils import str2bool, seed
from global_configs import *
from loss import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    choices=["mosi", "mosei, iemocap, meld"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=40)
parser.add_argument("--train_batch_size", type=int, default=48)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument("--num_heads", type=int, default=8)
parser.add_argument("--layers", type=int, default=4)
parser.add_argument("--attn_dropout", type=int, default=0.1)
parser.add_argument("--relu_dropout", type=float, default=0.1)
parser.add_argument("--res_dropout", type=float, default=0.1)
parser.add_argument("--embed_dropout", type=float, default=0.25)
parser.add_argument("--attn_mask", type=bool, default=True)
parser.add_argument("--fusion_dim", type=int, default=128)
parser.add_argument("--fusion_dropout", type=float, default=0.1)
parser.add_argument("--text_dropout", type=float, default=0.1)
parser.add_argument("--audio_dropout", type=float, default=0.1)
parser.add_argument("--video_dropout", type=float, default=0.1)
parser.add_argument("--out_dropout", type=float, default=0.1)
parser.add_argument("--post_text_dim", type=int, default=64)
parser.add_argument("--post_audio_dim", type=int, default=32)
parser.add_argument("--post_video_dim", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=seed, default="random")
parser.add_argument("--save_path", type=str, default="./checkpoint/convbert_TAV_0.05.pt")

pwd = os.path.abspath(__file__)
logging.basicConfig(
            filename=os.path.join(Path(pwd).parent, "model_DG.log"),
            format='%(asctime)s %(message)s',
            datefmt='%m/%d/%Y %p %I:%M:%S',
            level=logging.INFO
        )
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("trainLogger")
logger.addHandler(logging.StreamHandler(sys.stdout))

args = parser.parse_args()


class Multimodal_Datasets(Dataset):
    def __init__(self, meld, iemocap, mosi, mosei):
        super(Multimodal_Datasets, self).__init__()
        self.tokenizer = ConvBertTokenizer.from_pretrained('YituTech/conv-bert-base')
        self.mosi_text = mosi['text']
        self.mosi_video = mosi['video']
        self.mosi_audio = mosi['audio']
        self.mosi_label = mosi['label']
        self.mosi_aug = mosi['aug']
        self.mosei_text = mosei['text']
        self.mosei_video = mosei['video']
        self.mosei_audio = mosei['audio']
        self.mosei_label = mosei['label']
        self.iemocap_text = iemocap['text']
        self.iemocap_video = iemocap['video']
        self.iemocap_audio = iemocap['audio']
        self.iemocap_label = iemocap['label']
        self.iemocap_aug = iemocap['aug']
        self.meld_text = meld['text']
        self.meld_video = meld['video']
        self.meld_audio = meld['audio']
        self.meld_label = meld['label']
        self.meld_aug = meld['aug']
        # self.mosi_text = mosi['text'][:100]
        # self.mosi_video = mosi['video'][:100]
        # self.mosi_audio = mosi['audio'][:100]
        # self.mosi_label = mosi['label'][:100]
        # self.mosi_aug = mosi['aug'][:100]
        # self.mosei_text = mosei['text'][:100]
        # self.mosei_video = mosei['video'][:100]
        # self.mosei_audio = mosei['audio'][:100]
        # self.mosei_label = mosei['label'][:100]
        # self.iemocap_text = iemocap['text'][:100]
        # self.iemocap_video = iemocap['video'][:100]
        # self.iemocap_audio = iemocap['audio'][:100]
        # self.iemocap_label = iemocap['label'][:100]
        # self.iemocap_aug = iemocap['aug'][:100]
        # self.meld_text = meld['text'][:100]
        # self.meld_video = meld['video'][:100]
        # self.meld_audio = meld['audio'][:100]
        # self.meld_label = meld['label'][:100]
        # self.meld_aug = meld['aug'][:100]

    def prepare_bert_input(self, text):
        input_ids = self.tokenizer.encode_plus(text[0])['input_ids']
        segment_ids = self.tokenizer.encode_plus(text[0])['token_type_ids']
        attention_mask = self.tokenizer.encode_plus(text[0])['attention_mask']
        if len(input_ids) < args.max_seq_length:
            input_ids.extend([0] * (args.max_seq_length - len(input_ids)))
            segment_ids.extend([0] * (args.max_seq_length - len(segment_ids)))
            attention_mask.extend([0] * (args.max_seq_length - len(attention_mask)))
        else:
            input_ids = input_ids[:args.max_seq_length]
            segment_ids = segment_ids[:args.max_seq_length]
            attention_mask = attention_mask[:args.max_seq_length]
        return np.array(input_ids), np.array(attention_mask), np.array(segment_ids)

    def __getitem__(self, index):
        meld_input_ids, meld_attention_mask, meld_segment_ids = self.prepare_bert_input(self.meld_text[index])
        iemocap_input_ids, iemocap_attention_mask, iemocap_segment_ids = self.prepare_bert_input(
            self.iemocap_text[index])
        mosi_input_ids, mosi_attention_mask, mosi_segment_ids = self.prepare_bert_input(self.mosi_text[index])
        mosei_input_ids, mosei_attention_mask, mosei_segment_ids = self.prepare_bert_input(self.mosei_text[index])
        input_dict = {
            'meld': {
                'audio': torch.tensor(self.meld_audio[index]),
                'vision': torch.tensor(self.meld_video[index]),
                'input_ids': torch.tensor(meld_input_ids),
                'segment_ids': torch.tensor(meld_segment_ids),
                'attention_mask': torch.tensor(meld_attention_mask),
                'annotations': torch.tensor(self.meld_label[index]),
                'aug': torch.tensor(self.meld_aug[index]),
            },
            'iemocap':
                {
                    'audio': torch.tensor(self.iemocap_audio[index]),
                    'vision': torch.tensor(self.iemocap_video[index]),
                    'input_ids': torch.tensor(iemocap_input_ids),
                    'segment_ids': torch.tensor(iemocap_segment_ids),
                    'attention_mask': torch.tensor(iemocap_attention_mask),
                    'annotations': torch.tensor(self.iemocap_label[index]),
                    'aug': torch.tensor(self.iemocap_aug[index]),
                },
            'mosi': {
                'audio': torch.tensor(self.mosi_audio[index]),
                'vision': torch.tensor(self.mosi_video[index]),
                'input_ids': torch.tensor(mosi_input_ids),
                'segment_ids': torch.tensor(mosi_segment_ids),
                'attention_mask': torch.tensor(mosi_attention_mask),
                'annotations': torch.tensor(self.mosi_label[index]),
                'aug': torch.tensor(self.mosi_aug[index]),
            },
            'mosei':
                {
                    'audio': torch.tensor(self.mosei_audio[index]),
                    'vision': torch.tensor(self.mosei_video[index]),
                    'input_ids': torch.tensor(mosei_input_ids),
                    'segment_ids': torch.tensor(mosei_segment_ids),
                    'attention_mask': torch.tensor(mosei_attention_mask),
                    'annotations': torch.tensor(self.mosei_label[index]),
                }
        }
        return input_dict

    def __len__(self):
        return len(self.mosi_text)


def set_up_data_loader(path):
    meld_f = open(path + "meld.pkl", "rb")
    iemocap_f = open(path + "iemocap.pkl", "rb")
    mosi_f = open(path + "mosi.pkl", "rb")
    mosei_f = open(path + "mosei.pkl", "rb")

    meld = pickle.load(meld_f)
    iemocap = pickle.load(iemocap_f)
    mosi = pickle.load(mosi_f)
    mosei = pickle.load(mosei_f)

    meld_train_data = meld["train"]
    meld_dev_data = meld["valid"]

    iemocap_train_data = iemocap["train"]
    iemocap_dev_data = iemocap["valid"]

    mosi_train_data = mosi["train"]
    mosi_dev_data = mosi["valid"]

    mosei_train_data = mosei["train"]
    mosei_dev_data = mosei["valid"]

    train_dataset = Multimodal_Datasets(meld_train_data, iemocap_train_data, mosi_train_data, mosei_train_data)
    dev_dataset = Multimodal_Datasets(meld_dev_data, iemocap_dev_data, mosi_dev_data, mosei_dev_data)

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


def prep_for_training(num_train_optimization_steps: int):
    multimodal_config = MultimodalConfig(
        beta_shift=args.beta_shift, dropout_prob=args.dropout_prob
    )
    model = MAG_ConvBertForSequenceClassification.from_pretrained(
        'YituTech/conv-bert-base', multimodal_config=multimodal_config,
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


def calc_loss(meld_outputs, iemocap_outputs, mosi_outputs, mosei_outputs, meld_label, iemocap_label, mosi_label, mosei_label):
    criterion_cls = nn.CrossEntropyLoss().cuda()
    criterion_reg = nn.MSELoss().cuda()
    meld_loss = criterion_cls(meld_outputs, meld_label)
    iemocap_loss = criterion_cls(iemocap_outputs, iemocap_label)
    mosi_loss = criterion_reg(mosi_outputs, mosi_label)
    mosei_loss = criterion_reg(mosei_outputs, mosei_label)
    return meld_loss, iemocap_loss, mosi_loss, mosei_loss


def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        meld_acoustic = batch['meld']['audio'].to(DEVICE)
        meld_visual = batch['meld']['vision'].to(DEVICE)
        meld_input_ids = batch['meld']['input_ids'].to(DEVICE)
        meld_segment_ids = batch['meld']['segment_ids'].to(DEVICE)
        meld_attention_mask = batch['meld']['attention_mask'].to(DEVICE)
        meld_label = batch['meld']['annotations'].to(DEVICE)
        meld_aug = batch['meld']['aug'].to(DEVICE)
        iemocap_acoustic = batch['iemocap']['audio'].to(DEVICE)
        iemocap_visual = batch['iemocap']['vision'].to(DEVICE)
        iemocap_input_ids = batch['iemocap']['input_ids'].to(DEVICE)
        iemocap_segment_ids = batch['iemocap']['segment_ids'].to(DEVICE)
        iemocap_attention_mask = batch['iemocap']['attention_mask'].to(DEVICE)
        iemocap_label = batch['iemocap']['annotations'].to(DEVICE)
        iemocap_aug = batch['iemocap']['aug'].to(DEVICE)
        mosi_acoustic = batch['mosi']['audio'].to(DEVICE)
        mosi_visual = batch['mosi']['vision'].to(DEVICE)
        mosi_input_ids = batch['mosi']['input_ids'].to(DEVICE)
        mosi_segment_ids = batch['mosi']['segment_ids'].to(DEVICE)
        mosi_attention_mask = batch['mosi']['attention_mask'].to(DEVICE)
        mosi_label = batch['mosi']['annotations'].squeeze(2).to(DEVICE)
        mosi_aug = batch['mosi']['aug'].to(DEVICE)
        mosei_acoustic = batch['mosei']['audio'].to(DEVICE)
        mosei_visual = batch['mosei']['vision'].to(DEVICE)
        mosei_input_ids = batch['mosei']['input_ids'].to(DEVICE)
        mosei_segment_ids = batch['mosei']['segment_ids'].to(DEVICE)
        mosei_attention_mask = batch['mosei']['attention_mask'].to(DEVICE)
        mosei_label = batch['mosei']['annotations'].squeeze(2).to(DEVICE)
        meld_logits, _, meld_outputs = model(meld_input_ids, meld_acoustic.float(), meld_visual, meld_attention_mask,
                                          meld_segment_ids, dataset='meld', aug=meld_aug)
        iemocap_logits, _, iemocap_outputs = model(iemocap_input_ids, iemocap_acoustic.float(), iemocap_visual,
                                                iemocap_attention_mask, iemocap_segment_ids, dataset='iemocap', aug=iemocap_aug)
        mosi_logits, _, mosi_outputs = model(mosi_input_ids, mosi_acoustic.float(), mosi_visual,
                                          mosi_attention_mask, mosi_segment_ids, dataset='mosi', aug=mosi_aug)
        mosei_logits, _, mosei_outputs = model(mosei_input_ids, mosei_acoustic.float(), mosei_visual, mosei_attention_mask,
                                            mosei_segment_ids, dataset='mosei')
        # loss_msda = 0.0005 * msda_regulizer(meld_outputs, iemocap_outputs,
        #                                       mosei_outputs, mosi_outputs, 5)
        loss_msda = 0.05 * msda_regulizer(meld_outputs, iemocap_outputs,
                                            mosei_outputs, mosi_outputs, 5)

        meld_loss, iemocap_loss, mosi_loss, mosei_loss = calc_loss(meld_logits, iemocap_logits, mosi_logits, mosei_logits,
                              meld_label, iemocap_label, mosi_label.float(), mosei_label)

        loss = loss_msda + meld_loss + iemocap_loss + mosi_loss + mosei_loss
        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        loss.backward()

        tr_loss += loss.item()
        nb_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    print("msda loss: {}".format(loss_msda))
    print("meld loss: {}\tiemocap loss: {}\tmosi loss: {}\tmosei loss: {}".format(meld_loss, iemocap_loss, mosi_loss,
                                                                                  mosei_loss))

    return tr_loss / nb_tr_steps


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            meld_acoustic = batch['meld']['audio'].to(DEVICE)
            meld_visual = batch['meld']['vision'].to(DEVICE)
            meld_input_ids = batch['meld']['input_ids'].to(DEVICE)
            meld_segment_ids = batch['meld']['segment_ids'].to(DEVICE)
            meld_attention_mask = batch['meld']['attention_mask'].to(DEVICE)
            meld_label = batch['meld']['annotations'].to(DEVICE)
            iemocap_acoustic = batch['iemocap']['audio'].to(DEVICE)
            iemocap_visual = batch['iemocap']['vision'].to(DEVICE)
            iemocap_input_ids = batch['iemocap']['input_ids'].to(DEVICE)
            iemocap_segment_ids = batch['iemocap']['segment_ids'].to(DEVICE)
            iemocap_attention_mask = batch['iemocap']['attention_mask'].to(DEVICE)
            iemocap_label = batch['iemocap']['annotations'].to(DEVICE)
            mosi_acoustic = batch['mosi']['audio'].to(DEVICE)
            mosi_visual = batch['mosi']['vision'].to(DEVICE)
            mosi_input_ids = batch['mosi']['input_ids'].to(DEVICE)
            mosi_segment_ids = batch['mosi']['segment_ids'].to(DEVICE)
            mosi_attention_mask = batch['mosi']['attention_mask'].to(DEVICE)
            mosi_label = batch['mosi']['annotations'].squeeze(2).to(DEVICE)
            mosei_acoustic = batch['mosei']['audio'].to(DEVICE)
            mosei_visual = batch['mosei']['vision'].to(DEVICE)
            mosei_input_ids = batch['mosei']['input_ids'].to(DEVICE)
            mosei_segment_ids = batch['mosei']['segment_ids'].to(DEVICE)
            mosei_attention_mask = batch['mosei']['attention_mask'].to(DEVICE)
            mosei_label = batch['mosei']['annotations'].squeeze(2).to(DEVICE)
            meld_logits, _, meld_outputs = model(meld_input_ids, meld_acoustic.float(), meld_visual,
                                                 meld_attention_mask,
                                                 meld_segment_ids, dataset='meld')
            iemocap_logits, _, iemocap_outputs = model(iemocap_input_ids, iemocap_acoustic.float(), iemocap_visual,
                                                       iemocap_attention_mask, iemocap_segment_ids, dataset='iemocap')
            mosi_logits, _, mosi_outputs = model(mosi_input_ids, mosi_acoustic.float(), mosi_visual,
                                                 mosi_attention_mask, mosi_segment_ids, dataset='mosi')
            mosei_logits, _, mosei_outputs = model(mosei_input_ids, mosei_acoustic.float(), mosei_visual,
                                                   mosei_attention_mask,
                                                   mosei_segment_ids, dataset='mosei')
            # loss_msda = 0.0005 * msda_regulizer(meld_outputs, iemocap_outputs,
            #                                     mosei_outputs, mosi_outputs, 5)
            loss_msda = 0.05 * msda_regulizer(meld_outputs, iemocap_outputs,
                                                mosei_outputs, mosi_outputs, 5)
            meld_loss, iemocap_loss, mosi_loss, mosei_loss = calc_loss(meld_logits, iemocap_logits, mosi_logits,
                                                                       mosei_logits,
                                                                       meld_label, iemocap_label, mosi_label.float(),
                                                                       mosei_label)

            loss = loss_msda + meld_loss + iemocap_loss + mosi_loss + mosei_loss

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
        isBetter = valid_loss <= (best_valid - 1e-6)
        if isBetter:
            best_valid, best_epoch = valid_loss, epoch_i
            torch.save({
                'epoch': epoch_i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, args.save_path)
            print("*****************MODEL SAVED*****************")

        print("epoch:{}, train_loss:{}, valid_loss:{}".format(epoch_i, train_loss, valid_loss))


def main():
    set_random_seed(2897)
    # set_random_seed(args.seed)
    logger.info("seed:{}".format(args.seed))

    (
        train_data_loader,
        dev_data_loader,
        num_train_optimization_steps,
    ) = set_up_data_loader(path="./datasets_new/data_aug/")

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
