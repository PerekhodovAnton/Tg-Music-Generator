import os
import mido
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from io import BytesIO

def midi_to_audio(midi_path, sample_path, output_mp3, tempo=120, sample_rate=44100):
    # Загрузка сэмпла
    sample_audio, sr = librosa.load(sample_path, sr=sample_rate)
    
    # Инициализация аудиобуфера
    audio_buffer = np.zeros(int(sample_rate * 60 * 10))  # 10 минут по умолчанию
    current_tempo = tempo
    
    # Парсинг MIDI
    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat
    time = 0.0  # в секундах
    
    for msg in mid:
        time += mido.tick2second(msg.time, ticks_per_beat, mido.bpm2tempo(current_tempo))
        
        if msg.type == 'set_tempo':
            current_tempo = mido.tempo2bpm(msg.tempo)
            
        if msg.type == 'note_on' and msg.velocity > 0:
            note = msg.note
            start_time = time
            
            # Поиск соответствующего сообщения note_off
            end_time = start_time
            track_time = time
            for track_msg in mid.tracks[0]:
                track_time += mido.tick2second(track_msg.time, ticks_per_beat, mido.bpm2tempo(current_tempo))
                if track_msg.type == 'note_off' and track_msg.note == note:
                    end_time = track_time
                    break
                elif track_msg.type == 'note_on' and track_msg.note == note and track_msg.velocity == 0:
                    end_time = track_time
                    break
            
            # Рассчет длительности ноты
            duration = end_time - start_time
            
            # Изменение высоты звука сэмпла
            sample_pitch = 60  # C4 (нота сэмпла по умолчанию)
            pitch_shift = note - sample_pitch
            shifted_audio = librosa.effects.pitch_shift(sample_audio, sr=sample_rate, n_steps=pitch_shift)
            
            # Наложение на аудиобуфер
            start_sample = int(start_time * sample_rate)
            end_sample = start_sample + len(shifted_audio)
            
            if end_sample > len(audio_buffer):
                audio_buffer = np.pad(audio_buffer, (0, end_sample - len(audio_buffer)))
            
            audio_buffer[start_sample:end_sample] += shifted_audio

    # Нормализация и сохранение в WAV
    audio_buffer /= np.max(np.abs(audio_buffer))
    with BytesIO() as wav_buffer:
        sf.write(wav_buffer, audio_buffer, sample_rate, format='WAV')
        wav_buffer.seek(0)
        
        # Конвертация в MP3
        audio = AudioSegment.from_wav(wav_buffer)
        audio.export(output_mp3, format='mp3', bitrate='192k')

# Использование
midi_to_audio(
    midi_path='input.mid',
    sample_path='piano_c4.wav',  # Сэмпл ноты C4
    output_mp3='output.mp3',
    tempo=120  # Желаемый темп (BPM)
)