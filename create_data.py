import os
import cv2
import glob
import pickle
import whisper
import librosa
from PIL import Image
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from transformers import BertTokenizerFast


class create_data(object):
    def __init__(self, data_path, split, data_num, max_len):
        self.data_path = data_path
        self.split = split
        self.max_len = max_len
        self.tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.mtcnn = MTCNN(image_size=160, margin=0)
        self.video_list = []
        for vid in data_num:
            for i, f in enumerate(glob.glob(os.path.join(self.data_path, vid, "movie/*.m2ts"))):
                self.video_list.append(f)
                # if i == 4:
                #     break
        for dir in ['mp4', 'wav', 'jpg', 'cropped']:
            if not os.path.exists(dir):
                os.makedirs(dir)
        with open(self.data_path + "sentiment_label.txt", "r") as f:
            self.contents = f.readlines()
        self.input_dict = {
            'audio': [],
            'video': [],
            'text': [],
            'label': []
        }

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
            vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 200))  # added this line
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
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=2048, n_mfcc=64)
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

    def output_data(self, selected_video):
        text, img_feat, audio_feat = self.extract_feat(selected_video)
        label = self.get_label(selected_video)
        return text, audio_feat, img_feat, label

    def create_dict(self):
        for video in self.video_list:
            text, audio_feat, img_feat, label = self.output_data(video)
            self.input_dict['audio'].append(audio_feat)
            self.input_dict['video'].append(img_feat)
            self.input_dict['text'].append(text)
            self.input_dict['label'].append(torch.tensor([[label]]))
        data_dict = {self.split: self.input_dict}
        return data_dict


if __name__ == "__main__":
    data_path = "/home/yewon/ssd2/ai31/sentiment_analysis/Korean/audiotextvision-transformer/data/korean_multimodal_dataset/"
    train_num = ['003', '006', '007', '015', '018', '040']
    val_num = ['041']
    train_data = create_data(data_path, split='train', data_num=train_num, max_len=40)
    train_dict = train_data.create_dict()
    with open('train.pkl', 'wb') as handle:
        pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    val_data = create_data(data_path, split='val', data_num=val_num, max_len=40)
    val_dict = val_data.create_dict()
    with open('val.pkl', 'wb') as handle:
        pickle.dump(val_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
