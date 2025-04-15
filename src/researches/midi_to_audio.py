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
    # Предполагаем, что сэмпл соответствует ноте C4 (MIDI 60)
    sample_pitch = 60

    # Инициализация аудиобуфера (например, 1 минута)
    audio_buffer = np.zeros(int(sample_rate * 60 * 1))
    
    # Загрузка MIDI
    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat
    # Начальный темп (в микросекундах на бит) для заданного BPM
    current_tempo = mido.bpm2tempo(tempo)
    current_time = 0.0  # Текущее время в секундах

    # Используем объединённый трек, чтобы корректно обработать аккорды (одновременные сообщения)
    merged_track = mido.merge_tracks(mid.tracks)
    
    # Словарь для активных нот: ключ — номер ноты, значение — список времен начала
    active_notes = {}
    # Список событий вида (номер ноты, время начала, время окончания)
    note_events = []
    
    for msg in merged_track:
        # Вычисляем прошедшее время для данного сообщения с учётом текущего темпа
        delta = mido.tick2second(msg.time, ticks_per_beat, current_tempo)
        current_time += delta
        
        # Обновление темпа по событию
        if msg.type == 'set_tempo':
            current_tempo = msg.tempo
        
        # Обработка запуска ноты
        elif msg.type == 'note_on' and msg.velocity > 0:
            # Если для данной ноты уже есть одно или несколько событий, добавляем еще время начала
            active_notes.setdefault(msg.note, []).append(current_time)
        
        # Обработка окончания ноты (note_off или note_on с velocity=0)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in active_notes and active_notes[msg.note]:
                start_time = active_notes[msg.note].pop(0)
                end_time = current_time
                note_events.append((msg.note, start_time, end_time))
    
    # Обработка каждого события ноты
    for note, start_time, end_time in note_events:
        duration = end_time - start_time  # длительность в секундах
        if duration <= 0:
            continue

        # Считаем изменение высоты (pitch shift) относительно сэмпла C4
        pitch_shift = note - sample_pitch
        shifted_audio = librosa.effects.pitch_shift(sample_audio, sr=sample_rate, n_steps=pitch_shift)
        
        # Определяем длительность сдвинутого сэмпла
        sample_duration = len(shifted_audio) / sample_rate

        # Если сэмпл дольше, чем требуется, обрезаем его до нужной длительности
        if sample_duration > duration:
            target_length = int(duration * sample_rate)
            processed_audio = shifted_audio[:target_length]
        else:
            # Если сэмпл короче, растягиваем (time stretching)
            rate = sample_duration / duration
            processed_audio = librosa.effects.time_stretch(shifted_audio, rate=rate)
            target_length = int(duration * sample_rate)
            if len(processed_audio) > target_length:
                processed_audio = processed_audio[:target_length]
            else:
                processed_audio = np.pad(processed_audio, (0, target_length - len(processed_audio)))
        
        # Расчёт позиции вставки в аудиобуфер
        start_sample = int(start_time * sample_rate)
        end_sample = start_sample + len(processed_audio)
        
        if end_sample > len(audio_buffer):
            audio_buffer = np.pad(audio_buffer, (0, end_sample - len(audio_buffer)))
        
        # Накладываем аудиофрагмент: суммирование сигналов для поддержки аккордов
        audio_buffer[start_sample:end_sample] += processed_audio

    # Финальная нормализация аудио
    if np.max(np.abs(audio_buffer)) > 0:
        audio_buffer = audio_buffer / np.max(np.abs(audio_buffer))
    
    # Сохранение результата: сначала в WAV, затем конвертация в MP3
    with BytesIO() as wav_buffer:
        sf.write(wav_buffer, audio_buffer, sample_rate, format='WAV')
        wav_buffer.seek(0)
        audio = AudioSegment.from_wav(wav_buffer)
        audio.export(output_mp3, format='mp3', bitrate='192k')

# Пример использования
midi_to_audio(
    midi_path='/Users/anper/Documents/MusicGenerator/Tg-Music-Generator/src/midi/A - I V I IV.mid',
    sample_path='/Users/anper/Documents/MusicGenerator/Tg-Music-Generator/src/wav_samples/VR_synth_note_drifter_C.wav',  # сэмпл C4
    output_mp3='output.mp3',
    tempo=120  # начальный темп (BPM)
)
