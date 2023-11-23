import os
import cv2
import glob
import whisper
import librosa
from PIL import Image
import numpy as np
import torch
import torchvision
from torch.utils.data.dataset import Dataset
from facenet_pytorch import MTCNN, InceptionResnetV1
from transformers import BertTokenizerFast, BertModel


class create_dataset(Dataset):
    def __init__(self, data_path, data_num):
        super(create_dataset, self).__init__()
        self.data_path = data_path
        self.max_len = 40
        self.tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.mtcnn = MTCNN(image_size=160, margin=0)
        self.video_list = []
        for vid in data_num:
            for f in glob.glob(os.path.join(self.data_path, vid, "movie/*.m2ts")):
                self.video_list.append(f)
        for dir in ['mp4', 'wav', 'jpg', 'cropped']:
            if not os.path.exists(dir):
                os.makedirs(dir)
        with open(self.data_path + "sentiment_label.txt", "r") as f:
            self.contents = f.readlines()

    def extract_feat(self, file):
        mts_f = file.split("/")[-1]
        mp4_f = './mp4/' + mts_f.split('.')[0] + '.mp4'
        wav_f = './wav/' + mts_f.split('.')[0] + '.wav'
        self.mts2mp4(mp4_f, file)
        self.video2wav(mp4_f, wav_f)
        text = self.stt(wav_f)
        img_feat = self.vid2img_embed(mp4_f)
        audio_feat = self.audio_FE(wav_f)
        return text, img_feat, audio_feat

    def mts2mp4(self, mp4_file, mts_file):
        threads = 4
        os.system("ffmpeg -i %s -threads %d -f mp4 %s" % (mts_file, threads, mp4_file))
        print("completed converting mts to mp4 for %s" % mts_file)
        # if os.path.exists(new_filename):
        #     os.remove(old_filename)

    def video2wav(self, mp4_file, wav_file):
        os.system("ffmpeg -i %s %s" % (mp4_file, wav_file))
        print("completed converting mp4 to wav for %s" % mp4_file)
        # if os.path.exists(new_filename):
        #     os.remove(old_filename)

    def stt(self, wav_file):
        model = whisper.load_model("base")

        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(wav_file)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # detect the spoken language
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)

        return result.text

    def vid2img_embed(self, path):
        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()
        count = 0
        img_feat = []
        while success:
            success, image = vidcap.read()
            print('Read a new frame: ', success)
            if success is False:
                break
            vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 500))  # added this line
            cv2.imwrite("./jpg/frame%d.jpg" % count, image)  # save frame as JPEG file
            img = Image.open("./jpg/frame%d.jpg" % count)
            img_cropped = self.mtcnn(img, save_path="./cropped/cropped_frame%d.jpg" % count)
            img_embedding = self.resnet(img_cropped.unsqueeze(0)) # unsqueeze to add batch dimension
            img_feat.extend(img_embedding.tolist())
            count += 1
        os.system("rm -r ./jpg/* ./cropped/* ./mp4/*")
        return torch.tensor(img_feat)

    def audio_FE(self, path):
        y, sr = librosa.load(path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=64)
        os.system("rm -r ./wav/*")
        return torch.tensor(mfcc).permute(1, 0)  # shape: (n_mfcc, t)

    def get_label(self, file):
        num = int(file.split("-")[-1].split(".")[0])
        label = float(self.contents[num-1].split(" ")[-1].strip())
        return label

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

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        selected_video = self.video_list[idx]
        text, img_feat, audio_feat = self.extract_feat(selected_video)
        input_ids, attention_mask, segment_ids = self.prepare_bert_input(text)
        label = self.get_label(selected_video)
        input_dict = {
            'audio': audio_feat,
            'video': img_feat,
            'input_ids': torch.tensor(input_ids),
            'segment_ids': torch.tensor(segment_ids),
            'attention_mask': torch.tensor(attention_mask),
            'label': torch.tensor([[label]])
        }
        return input_dict


if __name__ == "__main__":
    train_data = create_dataset(data_path="/home/yewon/ssd2/ai31/sentiment_analysis/Korean/audiotextvision-transformer/data/korean_multimodal_dataset/")
    print(train_data[0])