import os
import mido
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from io import BytesIO

def midi_to_audio_buffer(midi_path, sample_path, tempo=120, sample_rate=44100):
    """
    Converts a single MIDI file to an audio buffer using the given sample.
    Assumes the sample corresponds to MIDI note C4 (60) and uses a fixed tempo.
    
    Returns:
        audio_buffer (np.array): The rendered audio signal.
        duration (float): Duration in seconds of the rendered track.
    """
    # Load sample audio (assumed to be C4)
    sample_audio, sr = librosa.load(sample_path, sr=sample_rate)
    sample_pitch = 60  # MIDI note for C4
    
    # Load MIDI file
    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat
    # Use a fixed tempo (global BPM) regardless of any set_tempo events in the file.
    fixed_tempo = mido.bpm2tempo(tempo)
    current_time = 0.0  # In seconds
    
    # Merge all tracks to properly handle simultaneous events (like chords)
    merged_track = mido.merge_tracks(mid.tracks)
    
    # Dictionary for active notes and list for note events
    active_notes = {}
    note_events = []  # Each item is a tuple: (note, start_time, end_time)
    
    for msg in merged_track:
        # Always use the fixed tempo when converting ticks to seconds
        delta = mido.tick2second(msg.time, ticks_per_beat, fixed_tempo)
        current_time += delta
        
        # Ignore any 'set_tempo' events so that all messages use our fixed_tempo
        if msg.type == 'set_tempo':
            continue
        
        # Handle note-on events (with non-zero velocity)
        if msg.type == 'note_on' and msg.velocity > 0:
            active_notes.setdefault(msg.note, []).append(current_time)
        
        # Handle note-off events (or note_on with zero velocity)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in active_notes and active_notes[msg.note]:
                start_time = active_notes[msg.note].pop(0)
                note_events.append((msg.note, start_time, current_time))
    
    # Determine total duration (last note-off time) for this track.
    if note_events:
        max_time = max(end for _, _, end in note_events)
    else:
        max_time = 60  # Default duration if no events were found
    
    # Create an audio buffer covering the entire duration
    audio_length = int(sample_rate * (max_time + 1))  # Extra second of buffer
    audio_buffer = np.zeros(audio_length)
    
    # Process each note event and mix into the buffer
    for note, start_time, end_time in note_events:
        duration = end_time - start_time
        if duration <= 0:
            continue

        # Calculate pitch shift relative to the sample (C4)
        pitch_shift = note - sample_pitch
        shifted_audio = librosa.effects.pitch_shift(sample_audio, sr=sample_rate, n_steps=pitch_shift)
        
        # Match the note's duration by either truncating or stretching the sample
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
        
        # Sum the processed audio into the overall buffer to allow overlapping notes
        audio_buffer[start_sample:end_sample] += processed_audio

    # Normalize the final audio
    if np.max(np.abs(audio_buffer)) > 0:
        audio_buffer = audio_buffer / np.max(np.abs(audio_buffer))
    
    return audio_buffer, len(audio_buffer) / sample_rate

def collect_midis_to_audio(sources, output_mp3, sample_rate=44100):
    """
    Processes multiple MIDI files (each with an associated sample sound) and mixes them
    into a single track that plays concurrently.
    
    Each MIDI is rendered to its own audio buffer and then all buffers are padded to the
    length of the longest one before being summed.
    
    Args:
        sources (list of dict): Each dict should contain:
            - 'midi_path' (str): Path to the MIDI file.
            - 'sample_path' (str): Path to the sample sound (assumed to be C4).
            - 'tempo' (int, optional): BPM to use when processing the MIDI. Default is 120.
        output_mp3 (str): Path to the final MP3 file.
        sample_rate (int): Sample rate for processing and final output.
    """
    audio_buffers = []
    max_length = 0
    
    # Process each MIDI source
    for src in sources:
        midi_path = src.get('midi_path')
        sample_path = src.get('sample_path')
        tempo = src.get('tempo', 120)
        
        print(f"Processing '{midi_path}' with sample '{sample_path}' at tempo {tempo} BPM.")
        buffer_audio, duration = midi_to_audio_buffer(midi_path, sample_path, tempo, sample_rate)
        audio_buffers.append(buffer_audio)
        max_length = max(max_length, len(buffer_audio))
    
    # Create a master buffer with the length of the longest track
    master_buffer = np.zeros(max_length)
    
    # Pad each buffer to the same length and sum them
    for audio in audio_buffers:
        if len(audio) < max_length:
            audio = np.pad(audio, (0, max_length - len(audio)))
        master_buffer += audio
    
    # Normalize the master mix to prevent clipping
    if np.max(np.abs(master_buffer)) > 0:
        master_buffer = master_buffer / np.max(np.abs(master_buffer))
    
    # Export the final mixed audio to MP3 via an in-memory WAV buffer
    with BytesIO() as wav_buffer:
        sf.write(wav_buffer, master_buffer, sample_rate, format='WAV')
        wav_buffer.seek(0)
        final_audio = AudioSegment.from_wav(wav_buffer)
        final_audio.export(output_mp3, format='mp3', bitrate='192k')
    
    print(f"Final mix saved as '{output_mp3}'")

# Example usage:
sources = [
    {
        'midi_path': '/Users/anper/Documents/MusicGenerator/Tg-Music-Generator/src/midi/chords.mid',
        'sample_path': '/Users/anper/Documents/MusicGenerator/Tg-Music-Generator/src/wav_samples/VR_synth_note_drifter_C.wav',
        'tempo': 120
    },
    {
        'midi_path': '/Users/anper/Documents/MusicGenerator/Tg-Music-Generator/src/midi/melody.mid',
        'sample_path': '/Users/anper/Documents/MusicGenerator/Tg-Music-Generator/src/wav_samples/VR_synth_note_drifter_C.wav',
        'tempo': 120
    },
    # Add more sources if needed.
]

collect_midis_to_audio(sources, output_mp3='combined_output.mp3')
