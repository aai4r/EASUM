import os
import cv2
import whisper
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


def convert2mp4(path, mp4_file, mts_file):
    threads = 4
    os.chdir(path)
    os.system("ffmpeg -i %s -threads %d -f mp4 %s" % (mts_file, threads, mp4_file))
    print("ffmpeg completed converting for %s" % mts_file)
    # if os.path.exists(new_filename):
    #     os.remove(old_filename)


def video2wav(path, mp4_file, wav_file):
    os.chdir(path)
    os.system("ffmpeg -i %s %s" % (mp4_file, wav_file))
    print("ffmpeg completed converting for %s" % mp4_file)


def stt(wav_file):
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

    # print the recognized text
    print(result.text)


def vid2image(path):
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    count = 0
    while success:
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        if success is False:
            break
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 500))  # added this line
        cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
        count += 1


def video_FE(path):
    mtcnn = MTCNN(image_size=160, margin=0)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    img = Image.open(path)
    img_cropped = mtcnn(img, save_path='./test.jpg')
    img_embedding = resnet(img_cropped.unsqueeze(0))
    print(img_embedding.shape)


def crop(vid, i, j, h, w):
    return vid[..., i:(i + h), j:(j + w)]


def center_crop(vid, output_size):
    h, w = vid.shape[-2:]
    th, tw = output_size

    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(vid, i, j, th, tw)


def normalize(vid, mean, std):
    shape = (-1,) + (1,) * (vid.dim() - 1)
    mean = torch.as_tensor(mean).reshape(shape)
    std = torch.as_tensor(std).reshape(shape)
    return (vid - mean) / std


def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return center_crop(vid, self.size)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, vid):
        return normalize(vid, self.mean, self.std)


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        # NOTE: for those functions, which generally expect mini-batches, we keep them
        # as non-minibatch so that they are applied as if they were 4d (thus image).
        # this way, we only apply the transformation in the spatial domain
        interpolation = 'bilinear'
        # NOTE: using bilinear interpolation because we don't work on minibatches
        # at this level
        scale = None
        if isinstance(self.size, int):
            scale = float(self.size) / min(vid.shape[-2:])
            size = None
        else:
            size = self.size
        return torch.nn.functional.interpolate(
            vid, size=size, scale_factor=scale, mode=interpolation, align_corners=False,
            recompute_scale_factor=False
        )


class ToFloatTensorInZeroOne(object):
    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


if __name__ == "__main__":
    # path="/home/yewon/ssd2/ai31/sentiment_analysis/Korean/audiotextvision-transformer/data/korean_multimodal_dataset/006/movie"
    path = "/home/yewon/ssd2/ai31/sentiment_analysis/Korean/audiotextvision-transformer/data/korean_multimodal_dataset/000/picture/000-ang-00.JPG"
    video_FE(path)
    convert2mp4(path, './mp4/test.mp4', '006-004.m2ts')
    vid2image(path+"/mp4/test.mp4")
    video2wav(path, './mp4/test.mp4', './wav/test.wav')
    stt("./news_data/wav_files/SPK073/SPK073KBSCU058/SPK073KBSCU058M001.wav")


