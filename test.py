from __future__ import absolute_import, division, print_function

import argparse
import os
import random
import logging
import sys
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from bert import MAG_BertForSequenceClassification

from argparse_utils import seed
from global_configs import DEVICE
from test_loader import create_dataset

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


class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob


def set_up_data_loader(data_path):
    test_dataset = create_dataset(data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
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
    model.load_state_dict(torch.load(path))
    model.to(DEVICE)
    return model


def test(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):
            acoustic = batch['audio'].to(DEVICE)
            visual = batch['video'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            segment_ids = batch['segment_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            outputs = model(input_ids, visual, acoustic.float(), attention_mask, segment_ids)
            logits = outputs[0]

            logits = logits.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()

            preds.extend(logits)

        preds = np.array(preds)
        print("prediction:{}".format(preds))


def main(data_path, model_path):
    set_random_seed(args.seed)
    logger.info("seed:{}".format(args.seed))
    test_data_loader = set_up_data_loader(data_path=data_path)
    model = load_model(path=model_path)
    test(model, test_data_loader)


if __name__ == "__main__":
    main(data_path="./test_videos", model_path="./checkpoint/model_old.pt")

