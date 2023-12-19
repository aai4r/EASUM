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
        self.pos_vids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                         27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                         53, 63, 65, 70, 72, 74, 80, 81, 82, 83, 87, 88, 91, 94, 100, 120, 163, 303]
        self.data_path = data_path
        self.split = split
        self.max_len = max_len
        self.tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.mtcnn = MTCNN(image_size=160, margin=0)
        self.video_list = []
        # for vid in data_num:
        #     for i, f in enumerate(glob.glob(os.path.join(data_path, vid, "movie/*.m2ts"))):
        #         if int(f.split("-")[-1].split(".")[0]) in self.pos_vids:
        #             self.video_list.append(f)
        #             if len(self.video_list) >= 235:
        #                 break
        for i, vid in enumerate(data_num):
            for j, f in enumerate(glob.glob(os.path.join(data_path, vid, "movie/*.m2ts"))):
                if i > 0:
                    if int(f.split("-")[-1].split(".")[0]) in self.pos_vids:
                        self.video_list.append(f)
                else:
                    self.video_list.append(f)
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
    # vid_list = ['009', '010', '012', '013', '016', '017', '019', '020', '021', '023', '024', '027']
    vid_list = ['028', '030', '031']
    train_data = create_data(data_path, split='val', data_num=vid_list, max_len=40)
    train_dict = train_data.create_dict()
    with open('val.pkl', 'wb') as handle:
        pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

