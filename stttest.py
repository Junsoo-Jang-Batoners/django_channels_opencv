import speech_recognition as sr
import librosa

r = sr.Recognizer()

# sample_wav, rate = librosa.core.load('D:\[train] 음성데이터_wav/EX45RB113_EX0355_20210826.wav')

korean_audio = sr.AudioFile('test.wav')

with korean_audio as source:
    audio = r.record(source)
print(r.recognize_google(audio_data=audio, language='ko-KR'))
