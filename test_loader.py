import os
import cv2
import glob
import whisper
import librosa
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from facenet_pytorch import MTCNN, InceptionResnetV1
from transformers import BertTokenizerFast


class create_dataset(Dataset):
    def __init__(self, data_path):
        super(create_dataset, self).__init__()
        self.data_path = data_path
        self.max_len = 40
        self.tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.mtcnn = MTCNN(image_size=160, margin=0, thresholds=[0.3, 0.3, 0.3])
        self.video_list = []
        for f in glob.glob(os.path.join(self.data_path, "*")):
            self.video_list.append(f)
        for dir in ['wav', 'jpg', 'cropped']:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def extract_feat(self, file):
        img_feat = self.vid2img_embed(file)
        file_name = file.split("/")[-1]
        wav_f = './wav/' + file_name.split('.')[0] + '.wav'
        self.video2wav(file, wav_f)
        text = self.stt(wav_f)
        audio_feat = self.audio_FE(wav_f)
        return text, img_feat, audio_feat

    def video2wav(self, mp4_file, wav_file):
        os.system("ffmpeg -i %s %s" % (mp4_file, wav_file))
        print("completed converting mp4 to wav for %s" % mp4_file)

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
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=2048, n_mfcc=64)
        os.system("rm -r ./wav/*")
        return torch.tensor(mfcc).permute(1, 0)  # shape: (n_mfcc, t)

    def prepare_bert_input(self, text):
        input_ids = self.tokenizer.encode_plus(text)['input_ids']
        segment_ids = self.tokenizer.encode_plus(text)['token_type_ids']
        attention_mask = self.tokenizer.encode_plus(text)['attention_mask']
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

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        selected_video = self.video_list[idx]
        text, img_feat, audio_feat = self.extract_feat(selected_video)
        input_ids, attention_mask, segment_ids = self.prepare_bert_input(text)
        video, audio = self.collate(img_feat), self.collate(audio_feat)
        input_dict = {
            'audio': audio,
            'video': video,
            'input_ids': torch.tensor(input_ids),
            'segment_ids': torch.tensor(segment_ids),
            'attention_mask': torch.tensor(attention_mask),
        }
        return input_dict


if __name__ == "__main__":
    train_data = create_dataset(data_path="./test_videos")
    print(train_data[1])