from src.midi_to_audio import *

# Example usage:
sources = [
    {
        'midi_path': 'src/midi_keys/A/chords.mid',
        'sample_path': 'src/wav_samples/VR_synth_note_drifter_C.wav',
        'tempo': 120,
        'pitch': 60 # C4, every 12 notes - octave
    },
    {
        'midi_path': 'src/midi_keys/A/melody.mid',
        'sample_path': 'src/wav_samples/VEDH2 Synth Cut 009 C.wav',
        'tempo': 120,
        'pitch': 48
    },
    {
        'midi_path': 'src/midi_keys/A/bass.mid',
        'sample_path': 'src/wav_samples/VEDH2 Synth Cut 009 C.wav',
        'tempo': 120,
        'pitch': 84 #C2
    },
    
]

collect_midis_to_audio(sources, output_mp3='combined_output.mp3')