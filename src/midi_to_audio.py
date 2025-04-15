import os
import mido
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from io import BytesIO

def midi_to_audio_buffer(midi_path, sample_path, tempo=120, sample_rate=44100, base_pitch=60):
    """
    Converts a single MIDI file to an audio buffer using the given sample.
    The sample is assumed to correspond to a specific MIDI note (base_pitch).
    Uses a fixed tempo (ignoring internal tempo changes) to ensure consistency.

    Args:
        midi_path (str): Path to the MIDI file.
        sample_path (str): Path to the sample audio file.
        tempo (int): Global BPM to use when converting ticks to seconds.
        sample_rate (int): The sample rate for processing audio.
        base_pitch (int): The MIDI note number that the sample represents (default is 60 for C4).

    Returns:
        audio_buffer (np.array): The rendered audio signal.
        duration (float): Duration in seconds of the rendered track.
    """
    # Load sample audio (assumed to represent the note defined by base_pitch)
    sample_audio, sr = librosa.load(sample_path, sr=sample_rate)
    
    # Use the provided base_pitch instead of assuming C4 (60)
    sample_pitch = base_pitch  
    
    # Load MIDI file
    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat
    # Use a fixed tempo regardless of any set_tempo events within the MIDI
    fixed_tempo = mido.bpm2tempo(tempo)
    current_time = 0.0  # In seconds
    
    # Merge tracks to handle simultaneous events (chords)
    merged_track = mido.merge_tracks(mid.tracks)
    
    active_notes = {}
    note_events = []  # Will store tuples of (note, start_time, end_time)
    
    for msg in merged_track:
        # Always use fixed_tempo in conversion
        delta = mido.tick2second(msg.time, ticks_per_beat, fixed_tempo)
        current_time += delta
        
        # Ignore any tempo changes from the MIDI file
        if msg.type == 'set_tempo':
            continue
        
        if msg.type == 'note_on' and msg.velocity > 0:
            active_notes.setdefault(msg.note, []).append(current_time)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in active_notes and active_notes[msg.note]:
                start_time = active_notes[msg.note].pop(0)
                note_events.append((msg.note, start_time, current_time))
    
    # Determine track duration based on last event, or default to 60s
    if note_events:
        max_time = max(end for _, _, end in note_events)
    else:
        max_time = 60

    # Create an audio buffer covering the entire duration
    audio_length = int(sample_rate * (max_time + 1))
    audio_buffer = np.zeros(audio_length)
    
    # Process each note event
    for note, start_time, end_time in note_events:
        duration = end_time - start_time
        if duration <= 0:
            continue
        
        # Calculate the required pitch shift based on the specified base_pitch
        pitch_shift = note - sample_pitch
        shifted_audio = librosa.effects.pitch_shift(sample_audio, sr=sample_rate, n_steps=pitch_shift)
        
        sample_duration = len(shifted_audio) / sample_rate
        if sample_duration > duration:
            target_length = int(duration * sample_rate)
            processed_audio = shifted_audio[:target_length]
        else:
            rate = sample_duration / duration
            processed_audio = librosa.effects.time_stretch(shifted_audio, rate=rate)
            target_length = int(duration * sample_rate)
            if len(processed_audio) > target_length:
                processed_audio = processed_audio[:target_length]
            else:
                processed_audio = np.pad(processed_audio, (0, target_length - len(processed_audio)))
        
        start_sample = int(start_time * sample_rate)
        end_sample = start_sample + len(processed_audio)
        if end_sample > len(audio_buffer):
            audio_buffer = np.pad(audio_buffer, (0, end_sample - len(audio_buffer)))
        
        audio_buffer[start_sample:end_sample] += processed_audio

    if np.max(np.abs(audio_buffer)) > 0:
        audio_buffer = audio_buffer / np.max(np.abs(audio_buffer))
    
    return audio_buffer, len(audio_buffer) / sample_rate

def collect_midis_to_audio(sources, output_mp3, sample_rate=44100):
    """
    Processes multiple MIDI files (each with an associated sample sound and a defined base pitch)
    and mixes them into a single track that plays concurrently.
    
    Each MIDI is rendered to its own audio buffer. All buffers are then padded to the
    length of the longest one and summed.

    Each source dictionary should include:
        - 'midi_path' (str): Path to the MIDI file.
        - 'sample_path' (str): Path to the sample sound (audio file).
        - 'tempo' (int, optional): BPM (default is 120).
        - 'pitch' (int, optional): Base MIDI pitch of the sample sound (default is 60, for C4).

    Args:
        sources (list of dict): List of source dictionaries.
        output_mp3 (str): Path for the exported MP3 file.
        sample_rate (int): Sample rate for processing and final output.
    """
    audio_buffers = []
    max_length = 0
    
    for src in sources:
        midi_path = src.get('midi_path')
        sample_path = src.get('sample_path')
        tempo = src.get('tempo', 120)
        pitch = src.get('pitch', 60)
        
        print(f"Processing '{midi_path}' with sample '{sample_path}' at tempo {tempo} BPM, base pitch {pitch}.")
        buffer_audio, duration = midi_to_audio_buffer(midi_path, sample_path, tempo, sample_rate, base_pitch=pitch)
        audio_buffers.append(buffer_audio)
        max_length = max(max_length, len(buffer_audio))
    
    # Create a master buffer matching the duration of the longest track
    master_buffer = np.zeros(max_length)
    for audio in audio_buffers:
        if len(audio) < max_length:
            audio = np.pad(audio, (0, max_length - len(audio)))
        master_buffer += audio
    
    if np.max(np.abs(master_buffer)) > 0:
        master_buffer = master_buffer / np.max(np.abs(master_buffer))
    
    with BytesIO() as wav_buffer:
        sf.write(wav_buffer, master_buffer, sample_rate, format='WAV')
        wav_buffer.seek(0)
        final_audio = AudioSegment.from_wav(wav_buffer)
        final_audio.export(output_mp3, format='mp3', bitrate='192k')
    
    print(f"Final mix saved as '{output_mp3}'")
