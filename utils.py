import os
import whisper


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


if __name__ == "__main__":
    stt("./news_data/wav_files/SPK073/SPK073KBSCU058/SPK073KBSCU058M001.wav")


