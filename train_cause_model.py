import csv
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from transformers import BertTokenizer, BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained("beomi/kcbert-base")
tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


"""
여기 커피가 너무 맛있다 너무 고소해
여기 책상이 너무 더러운 데 우리 다른 데로 옮길까?
"""

class CustomDataset(Dataset):
    def __init__(self, file_name):
        self.input_file = csv.DictReader(open(file_name))
        self.tokenized_examples = self.prepare_data()

    def prepare_data(self):
        examples = {}
        examples['context'] = []
        examples['question'] = []
        examples['answer'] = []
        examples['answer_token'] = []
        for i, row in enumerate(self.input_file):
            examples['context'].append(row['context'])
            examples['question'].append(row['question'])
            examples['answer'].append(row['answer'])
            examples['answer_token'].append(tokenizer(examples['answer'][i])['input_ids'][1:-1])
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            max_length=256,
            truncation="only_second",
            padding="max_length"
        )
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        for i in range(len(tokenized_examples['input_ids'])):
            start, end = self.find_match(tokenized_examples["input_ids"][i], examples['answer_token'][i])
            tokenized_examples["start_positions"].append(start)
            tokenized_examples["end_positions"].append(end)

        return tokenized_examples

    def find_match(self, test_list, sublist):
        for idx in range(len(test_list) - len(sublist) + 1):
            if test_list[idx: idx + len(sublist)] == sublist:
                start = idx
                end = idx + len(sublist)
                break
        return start, end

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.tokenized_examples.items()}

    def __len__(self):
        return len(self.tokenized_examples.input_ids)


train_file_name = "food_review_train2.csv"
test_file_name = "food_review_test2.csv"
# train_dataset = CustomDataset(train_file_name)
test_dataset = CustomDataset(test_file_name)
PATH = './checkpoint/cause_model.pt'


def train_model(epochs):
    model.train()
    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    optim = AdamW(model.parameters(), lr=5e-5)
    outputs_model = []
    for epoch in range(epochs):
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            if len(outputs_model) == 0:
                outputs_model.append(outputs)
            loss = outputs[0]
            loss.backward()
            optim.step()
    torch.save(model.state_dict(), PATH)


def get_text_atrange(tokens, begin, end):
    answer = ""
    for i in range(begin, end + 1):
        if tokens[i][0:2] == '##': # If it's a subword token, then recombine it with the previous token.
            answer += tokens[i][2:]
        else: # Otherwise, add a space then the token.
            answer += ' ' + tokens[i]
    return answer


def test():
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for i, batch in enumerate(test_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # make predictions
            outputs = model(input_ids, attention_mask=attention_mask)
            # get top prediction with argmax
            start_pred = torch.argmax(outputs['start_logits'], dim=1)
            end_pred = torch.argmax(outputs['end_logits'], dim=1)
            cause_pred = tokenizer.decode(batch['input_ids'][0][start_pred:end_pred])
            print("Prediction: {}".format(cause_pred))
            print("\n")


def test_cause(input_ids, attention_mask):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        start_pred = torch.argmax(outputs['start_logits'], dim=1)
        end_pred = torch.argmax(outputs['end_logits'], dim=1)
        cause_pred = tokenizer.decode(input_ids[0][start_pred:end_pred])
        # print("Prediction: {}".format(cause_pred))
        # print("\n")
        return cause_pred


# train_model(epochs=3)
# test()





