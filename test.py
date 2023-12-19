from __future__ import absolute_import, division, print_function

import argparse
import os
import random
import logging
import sys
import numpy as np
from pathlib import Path
from PyQt5 import QtWidgets

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from bert import MAG_BertForSequenceClassification
from transformers import BertTokenizer, BertForQuestionAnswering

from utils import seed, Sentiment_Window
from global_configs import DEVICE
from test_loader import create_dataset
from train_cause_model import test_cause

parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_length", type=int, default=40)
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument("--seed", type=seed, default="random")

pwd = os.path.abspath(__file__)
logging.basicConfig(
            filename=os.path.join(Path(pwd).parent, "test_results.log"),
            format='%(asctime)s %(message)s',
            datefmt='%m/%d/%Y %p %I:%M:%S',
            level=logging.INFO
        )
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("testLogger")
logger.addHandler(logging.StreamHandler(sys.stdout))

args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")

class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob


def set_up_data_loader(data_path):
    test_dataset = create_dataset(data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    return test_dataloader


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
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


def load_model(path):
    multimodal_config = MultimodalConfig(
        beta_shift=args.beta_shift, dropout_prob=args.dropout_prob
    )

    model = MAG_BertForSequenceClassification.from_pretrained(
        'kykim/bert-kor-base', multimodal_config=multimodal_config, num_labels=1,
    )
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    return model


def test(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    app = QtWidgets.QApplication(sys.argv)
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):
            acoustic = batch['audio'].to(DEVICE)
            visual = batch['video'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            segment_ids = batch['segment_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            cause_input_ids = batch['cause_input_ids'].to(DEVICE)
            cause_attention_mask = batch['cause_attention_mask'].to(DEVICE)
            text = batch['text']
            print(text)
            outputs = model(input_ids, visual, acoustic.float(), attention_mask, segment_ids)
            logits = outputs[0]
            cause_pred = test_cause(cause_input_ids, attention_mask=cause_attention_mask)
            # start_pred = torch.argmax(cause_outputs['start_logits'], dim=1)
            # end_pred = torch.argmax(cause_outputs['end_logits'], dim=1)
            print("pred: {}".format(logits))
            if logits < 0:
                pred = "부정"
            elif logits == 0:
                pred = "중립"
            else:
                pred = "긍정"
            print("********Predicted Sentiment: {}********".format(pred))
            print("\n")
            # cause_pred = tokenizer.decode(batch['input_ids'][0][start_pred:end_pred])
            print("Cause Prediction: {}".format(cause_pred))
            mainWin = Sentiment_Window(sentiment=pred, cause=cause_pred)
            mainWin.show()


def main(data_path, model_path):
    # set_random_seed(args.seed)
    logger.info("seed:{}".format(args.seed))
    test_data_loader = set_up_data_loader(data_path=data_path)
    model = load_model(path=model_path)
    test(model, test_data_loader)


if __name__ == "__main__":
    main(data_path="./test_videos", model_path="./checkpoint/model.pt")

