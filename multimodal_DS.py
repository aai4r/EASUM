from __future__ import absolute_import, division, print_function

import argparse
import os
import random
import pickle
import logging
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from transformers import ConvBertTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from convbert import MAG_ConvBertForSequenceClassification_DS, MAG_ConvBertForSequenceClassification_DG, ConvBertForSequenceClassification

from argparse_utils import str2bool, seed
from global_configs import *
from loss import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    choices=["mosi", "mosei, iemocap, meld"], default="meld")
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
parser.add_argument("--seed", type=seed, default=2897)
# parser.add_argument("--seed", type=seed, default='random')
parser.add_argument("--save_path", type=str, default="./checkpoint/model_DS_meld_6c_2897.pt")

pwd = os.path.abspath(__file__)
logging.basicConfig(
            filename=os.path.join(Path(pwd).parent, "meld_6c_2897.log"),
            format='%(asctime)s %(message)s',
            datefmt='%m/%d/%Y %p %I:%M:%S',
            level=logging.INFO
        )
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("trainLogger")
logger.addHandler(logging.StreamHandler(sys.stdout))

args = parser.parse_args()


class Multimodal_Datasets(Dataset):
    def __init__(self, data):
        super(Multimodal_Datasets, self).__init__()
        self.tokenizer = ConvBertTokenizer.from_pretrained('YituTech/conv-bert-base')
        self.text = data['text']
        self.video = data['video']
        self.audio = data['audio']
        self.label = data['label']
        # self.text = data['text'][:100]
        # self.video = data['video'][:100]
        # self.audio = data['audio'][:100]
        # self.label = data['label'][:100]

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
        input_ids, attention_mask, segment_ids = self.prepare_bert_input(self.text[index])
        input_dict = {
            'audio': torch.tensor(self.audio[index]),
            'vision': torch.tensor(self.video[index]),
            'input_ids': torch.tensor(input_ids),
            'segment_ids': torch.tensor(segment_ids),
            'attention_mask': torch.tensor(attention_mask),
            'annotations': torch.tensor(self.label[index])
        }
        return input_dict

    def __len__(self):
        return len(self.text)


def set_up_data_loader(dataset):
    if dataset == 'meld':
        file = open("./datasets_new/data/meld_6.pkl", "rb")
    elif dataset == 'iemocap':
        file = open("./datasets_new/data/iemocap_6.pkl", "rb")
    elif dataset == 'mosi':
        file = open("./datasets_new/data/mosi.pkl", "rb")
    elif dataset == 'mosei':
        file = open("./datasets_new/data/mosei.pkl", "rb")

    data = pickle.load(file)

    train_data = data["train"]
    dev_data = data["valid"]
    test_data = data["test"]

    train_dataset = Multimodal_Datasets(train_data)
    dev_dataset = Multimodal_Datasets(dev_data)
    test_dataset = Multimodal_Datasets(test_data)

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

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True,
    )

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
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


def load_model(model_DS, model_DG):
    model_dict = model_DS.state_dict()
    pretrained_dict = model_DG.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model_DS.load_state_dict(model_dict)
    return model_DS


class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob


def prep_for_training(num_train_optimization_steps: int, load_path):
    multimodal_config = MultimodalConfig(
        beta_shift=args.beta_shift, dropout_prob=args.dropout_prob
    )
    model_DG = MAG_ConvBertForSequenceClassification_DG.from_pretrained(
        'YituTech/conv-bert-base', multimodal_config=multimodal_config,
    )
    # model_DG = ConvBertForSequenceClassification.from_pretrained(
    #     'YituTech/conv-bert-base'
    # )
    model_DS = MAG_ConvBertForSequenceClassification_DS.from_pretrained(
        'YituTech/conv-bert-base', multimodal_config=multimodal_config,
    )
    checkpoint = torch.load(load_path)
    model_DG.load_state_dict(checkpoint['model_state_dict'])
    model_DS = load_model(model_DS, model_DG)
    model_DG.to(DEVICE)
    model_DS.to(DEVICE)

    # Prepare optimizer
    model_param_optimizer = list(model_DS.named_parameters())
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
    return model_DS, model_DG, optimizer, scheduler


def calc_loss(logits, label, aux_logits=None, logits_DG=None):
    weight = 1
    aux_loss = 0
    if args.dataset == 'mosi' or args.dataset == 'mosei':
        label = label.squeeze(2)
        criterion = nn.MSELoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    loss = criterion(logits, label)
    if aux_logits is not None:
        if args.dataset == 'mosi' or args.dataset == 'mosei':
            aux_criterion = nn.KLDivLoss(reduction="batchmean")
            aux_loss = aux_criterion(aux_logits.log_softmax(dim=1), logits_DG.softmax(dim=1))
        else:
            aux_criterion = nn.MSELoss().cuda()
            aux_loss = aux_criterion(aux_logits, logits_DG)
    return loss + weight * aux_loss


def train_epoch(model_DS: nn.Module, model_DG: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    model_DS.train()
    model_DG.eval()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        acoustic = batch['audio'].to(DEVICE)
        visual = batch['vision'].to(DEVICE)
        input_ids = batch['input_ids'].to(DEVICE)
        segment_ids = batch['segment_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        label_ids = batch['annotations'].to(DEVICE)
        # logits_DG, outputs_DG = model_DG(input_ids, attention_mask, segment_ids, dataset=args.dataset, reverse=True)
        _, logits_DG, outputs_DG = model_DG(input_ids, acoustic.float(), visual, attention_mask, segment_ids,
                                         dataset=args.dataset, reverse=True)
        logits, aux_logits, _ = model_DS(input_ids, acoustic.float(), visual, attention_mask, segment_ids,
                                      dataset=args.dataset, DG_feat=outputs_DG)
        if args.dataset == 'mosi' or args.dataset == 'mosei':
            loss = calc_loss(logits, label_ids.float(), aux_logits, logits_DG)
        else:
            loss = calc_loss(logits, label_ids, aux_logits, logits_DG)
        # logits, aux_logits, _ = model_DS(input_ids, acoustic.float(), visual, attention_mask, segment_ids,
        #                               dataset=args.dataset)
        # loss = calc_loss(logits, label_ids.float())

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


def eval_epoch(model_DS: nn.Module, model_DG: nn.Module, dev_dataloader: DataLoader):
    model_DS.eval()
    model_DG.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            acoustic = batch['audio'].to(DEVICE)
            visual = batch['vision'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            segment_ids = batch['segment_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            label_ids = batch['annotations'].to(DEVICE)
            # logits_DG, outputs_DG = model_DG(input_ids, attention_mask, segment_ids, dataset=args.dataset, reverse=True)
            _, logits_DG, outputs_DG = model_DG(input_ids, acoustic.float(), visual, attention_mask, segment_ids,
                                             dataset=args.dataset, reverse=True)
            logits, aux_logits, _ = model_DS(input_ids, acoustic.float(), visual, attention_mask, segment_ids,
                                          dataset=args.dataset, DG_feat=outputs_DG)
            if args.dataset == 'mosi' or args.dataset == 'mosei':
                loss = calc_loss(logits, label_ids.float(), aux_logits, logits_DG)
            else:
                loss = calc_loss(logits, label_ids, aux_logits, logits_DG)
            # logits, aux_logits, _ = model_DS(input_ids, acoustic.float(), visual, attention_mask, segment_ids,
            #                               dataset=args.dataset)
            # loss = calc_loss(logits, label_ids.float())

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            nb_dev_steps += 1

    return dev_loss / nb_dev_steps


def test_epoch(model_DS: nn.Module, model_DG: nn.Module, test_dataloader: DataLoader):
    model_DS.eval()
    model_DG.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            acoustic = batch['audio'].to(DEVICE)
            visual = batch['vision'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            segment_ids = batch['segment_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            label_ids = batch['annotations'].to(DEVICE)
            # logits_DG, outputs_DG = model_DG(input_ids, attention_mask, segment_ids, dataset=args.dataset, reverse=True)
            _, logits_DG, outputs_DG = model_DG(input_ids, acoustic.float(), visual, attention_mask, segment_ids,
                                             dataset=args.dataset, reverse=True)
            logits, aux_logits, _ = model_DS(input_ids, acoustic.float(), visual, attention_mask, segment_ids,
                                          dataset=args.dataset, DG_feat=outputs_DG)
            # logits, aux_logits, _ = model_DS(input_ids, acoustic.float(), visual, attention_mask, segment_ids,
            #                               dataset=args.dataset)

            if args.dataset == 'mosi' or args.dataset == 'mosei':
                logits = logits.detach().cpu().numpy()
            else:
                logits = softmax(logits, dim=1)
                logits = logits.argmax(-1).detach().cpu().numpy()

            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels


def test_score_model(model: nn.Module, model_DG: nn.Module, test_dataloader: DataLoader, use_zero=False):

    preds, y_test = test_epoch(model, model_DG, test_dataloader)
    if args.dataset == 'mosi' or args.dataset == 'mosei':
        non_zeros = np.array(
            [i for i, e in enumerate(y_test) if e != 0 or False])
        zeros = np.array(
            [i for i, e in enumerate(y_test) if e != 0 or True])
        preds_non_zero = preds[non_zeros]
        y_test_non_zero = y_test[non_zeros]
        preds_zero = preds[zeros]
        y_test_zero = y_test[zeros]

        mae = np.mean(np.absolute(preds - y_test))
        corr = np.corrcoef(preds, y_test)[0][1]

        ##### binary #####
        preds_non_zero = preds_non_zero >= 0
        y_test_non_zero = y_test_non_zero >= 0

        preds_zero = preds_zero >= 0
        y_test_zero = y_test_zero >= 0

        f_score_non_zero = f1_score(y_test_non_zero, preds_non_zero, average="weighted")
        acc_non_zero = accuracy_score(y_test_non_zero, preds_non_zero)

        f_score_zero = f1_score(y_test_zero, preds_zero, average="weighted")
        acc_zero = accuracy_score(y_test_zero, preds_zero)

        return f_score_non_zero, acc_non_zero, f_score_zero, acc_zero, mae, corr
    else:
        acc = accuracy_score(y_test, preds)
        recall = recall_score(y_test, preds, average='weighted')
        precision = precision_score(y_test, preds, average='weighted')
        f1 = f1_score(y_test, preds, average='weighted')
        return acc, recall, precision, f1


def train(
    model_DS,
    model_DG,
    train_dataloader,
    validation_dataloader,
    test_data_loader,
    optimizer,
    scheduler,
):
    valid_losses = []
    test_accuracies = []

    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model_DS, model_DG, train_dataloader, optimizer, scheduler)
        valid_loss = eval_epoch(model_DS, model_DG, validation_dataloader)
        if args.dataset == 'mosi' or args.dataset == 'mosei':
            test_f1_non_zero, test_acc_non_zero, test_f1_zero, test_acc_zero, test_mae, test_corr = test_score_model(
                model_DS, model_DG, test_data_loader
            )

            logger.info(
                "epoch:{}, train_loss:{}, valid_loss:{}, test_f1_non_zero:{}, test_acc_non_zero:{}, test_f1_zero:{}, test_acc_zero:{}, test_mae:{}, test_corr:{}".format(
                    epoch_i, train_loss, valid_loss, test_f1_non_zero, test_acc_non_zero, test_f1_zero, test_acc_zero,
                    test_mae, test_corr
                )
            )
            valid_losses.append(valid_loss)
            test_accuracies.append(test_acc_non_zero)
            logger.info("best_test_acc:{}".format(max(test_accuracies)))
        else:
            acc, recall, precision, f1 = test_score_model(
                model_DS, model_DG, test_data_loader
            )

            logger.info(
                "epoch:{}, train_loss:{}, valid_loss:{}, test_f1:{}, test_acc:{}, test_precision:{}, test_recall:{}".format(
                    epoch_i, train_loss, valid_loss, f1, acc, precision, recall
                )
            )
            valid_losses.append(valid_loss)
            test_accuracies.append(acc)
            logger.info("best_test_acc:{}".format(max(test_accuracies)))
        if (epoch_i+1) % 10 == 0:
            torch.save({
                'epoch': epoch_i,
                'model_state_dict': model_DS.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, args.save_path)
            print("*****************MODEL SAVED*****************")


def main(dataset):
    # set_random_seed(2897)
    set_random_seed(args.seed)
    logger.info("seed:{}".format(args.seed))

    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_train_optimization_steps,
    ) = set_up_data_loader(dataset)

    model_DS, model_DG, optimizer, scheduler = prep_for_training(
        num_train_optimization_steps, load_path='./checkpoint/convbert_TAV_0.05.pt')

    train(
        model_DS,
        model_DG,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        optimizer,
        scheduler,
    )


if __name__ == "__main__":
    main(args.dataset)
