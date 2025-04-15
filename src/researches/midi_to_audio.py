import os
import mido
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from io import BytesIO
s = "ffmpeg.exe"
os.system(s)

AudioSegment.converter = f"{os.getcwd()}\\ffmpeg.exe"
AudioSegment.ffprobe = f"{os.getcwd()}\\ffprobe.exe"