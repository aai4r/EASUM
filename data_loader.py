import pickle
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast


train_batch_size, dev_batch_size = 2, 2
gradient_accumulation_step = 1
n_epochs = 5


class create_dataset(Dataset):
    def __init__(self, data):
        super(create_dataset, self).__init__()
        self.max_len = 40
        self.tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
        self.text = data['text']
        self.video = data['video']
        self.audio = data['audio']
        self.label = data['label']

    def prepare_bert_input(self, text):
        input_ids = self.tokenizer.encode_plus(text[0])['input_ids']
        segment_ids = self.tokenizer.encode_plus(text[0])['token_type_ids']
        attention_mask = self.tokenizer.encode_plus(text[0])['attention_mask']
        if len(input_ids) < self.max_len:
            input_ids.extend([0] * (self.max_len - len(input_ids)))
            segment_ids.extend([0] * (self.max_len - len(segment_ids)))
            attention_mask.extend([0] * (self.max_len - len(attention_mask)))
        else:
            input_ids = input_ids[:self.max_len]
            segment_ids = segment_ids[:self.max_len]
            attention_mask = attention_mask[:self.max_len]
        return np.array(input_ids), np.array(attention_mask), np.array(segment_ids)

    def collate(self, data):
        if len(data) < self.max_len:
            seq_len, dim = data.shape[0], data.shape[1]
            target = torch.zeros(self.max_len, dim)
            target[:seq_len, :] = data
            return target
        else:
            data = data[:self.max_len]
            return data

    def __getitem__(self, index):
        input_ids, attention_mask, segment_ids = self.prepare_bert_input(self.text[index])
        audio, video = self.collate(self.audio[index]), self.collate(self.video[index])
        input_dict = {
            'audio': audio,
            'video': video,
            'input_ids': torch.tensor(input_ids),
            'segment_ids': torch.tensor(segment_ids),
            'attention_mask': torch.tensor(attention_mask),
            'label': self.label[index]
        }
        return input_dict

    def __len__(self):
        return len(self.text)


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
                len(train_dataset) / train_batch_size /
                gradient_accumulation_step
            )
            * n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=dev_batch_size, shuffle=True
    )

    return (
        train_dataloader,
        dev_dataloader,
        num_train_optimization_steps,
    )


if __name__ == "__main__":
    (
        train_dataloader,
        dev_dataloader,
        num_train_optimization_steps,
    ) = set_up_data_loader()
    for step, batch in enumerate(train_dataloader):
        acoustic = batch['audio']
        visual = batch['video']
        input_ids = batch['input_ids']
        segment_ids = batch['segment_ids']
        attention_mask = batch['attention_mask']
        label_ids = batch['label']
        if step == 1:
            break