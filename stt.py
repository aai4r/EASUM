import os
import sys
import speech_recognition as sr


r = sr.Recognizer()
wav_path = "./news_data/wav_files/SPK073/SPK073KBSCU058"
for file_path in os.listdir(wav_path):
    kr_audio = sr.AudioFile(os.path.join(wav_path, file_path))

    with kr_audio as source:
        audio = r.record(source)

    print(file_path)
    print(r.recognize_google(audio, language='ko-KR'))
    # sys.stdout = open('news_out.txt', 'w')  # save text file