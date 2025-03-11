import os
import re
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
import music21
from music21 import stream, note, chord, tempo, key, meter

# Import the models from the models.py file
from models import SmallMusicGenerator, SmallMusicDiscriminator

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models_and_mappings():
    """
    Loads the model weights and the mapping files (note_to_int and int_to_note)
    from the 'requiredFiles' directory, and returns the models and the necessary data.
    """
    # Load the mappings
    with open('requiredFiles/note_to_int.pkl', 'rb') as f:
        note_to_int = pickle.load(f)
    with open('requiredFiles/int_to_note.pkl', 'rb') as f:
        int_to_note = pickle.load(f)

    n_vocab = len(note_to_int)
    sequence_length = 200  # You can adjust this based on what was used during training

    # Initialize the models
    generator = SmallMusicGenerator(n_vocab=n_vocab, sequence_length=sequence_length)
    discriminator = SmallMusicDiscriminator(n_vocab=n_vocab, sequence_length=sequence_length)

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    # Ensure positional encoding is on the correct device
    generator.positional_encoding = generator.positional_encoding.to(device)

    # Load the model weights
    checkpoint_path = "requiredFiles/best_model_Gloss_0.0319.pth"  # Update if the filename is different
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    # Set models to evaluation mode
    generator.eval()
    discriminator.eval()

    print("✅ Models and mappings loaded successfully.")
    return generator, discriminator, note_to_int, int_to_note, n_vocab, sequence_length

def generate_sequence(generator, seed_sequence, target_length, sequence_length, n_vocab, temperature=1.0):
    """
    Generates a long musical sequence using the generator.
    - seed_sequence: initial list of token indices
    - target_length: desired final length of the sequence (number of tokens)
    - temperature: controls the randomness (higher temperature yields more diverse output)
    """
    generator.eval()
    generated = seed_sequence.copy()
    while len(generated) < target_length:
        current_window = generated[-sequence_length:]
        input_seq = torch.tensor([current_window], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = generator(input_seq)  # [1, sequence_length, n_vocab]
        logits_last = logits[0, -1, :] / temperature
        probabilities = torch.softmax(logits_last, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1).item()
        generated.append(next_token)
    return generated

# Mapping from pitch class to note names
pitch_class_map = {
    0: "C", 1: "C#", 2: "D", 3: "D#",
    4: "E", 5: "F", 6: "F#", 7: "G",
    8: "G#", 9: "A", 10: "A#", 11: "B"
}

def parse_duration(dur_str, scale=1.0):
    """
    Converts a duration in 16th note units to music21's quarterLength.
    """
    try:
        dur = float(dur_str)
    except:
        dur = 1.0
    return (dur / 4.0) * scale

def token_to_music21(token):
    """
    Converts a token (NOTE_/CHORD_/REST_) to the corresponding music21 object.
    """
    if token.startswith("NOTE_"):
        parts = token.split("_")
        if len(parts) < 3:
            return None
        pitch_str = parts[1]
        dur = parse_duration(parts[2], scale=1.0)
        n_obj = note.Note(pitch_str)
        n_obj.quarterLength = dur
        return n_obj

    elif token.startswith("CHORD_"):
        parts = token.split("_")
        if len(parts) < 3:
            return None
        chord_numbers_str = parts[1]
        dur = parse_duration(parts[2], scale=1.0)
        numbers = chord_numbers_str.split(".")
        pitches = []
        for num_str in numbers:
            try:
                num = int(num_str)
                pc = num % 12
                pitch_name = pitch_class_map.get(pc, "C")
                pitches.append(f"{pitch_name}4")
            except:
                continue
        if pitches:
            c_obj = chord.Chord(pitches)
            c_obj.quarterLength = dur
            return c_obj
        else:
            return None

    elif token.startswith("REST_"):
        parts = token.split("_")
        if len(parts) < 2:
            return None
        dur = parse_duration(parts[1], scale=1.0)
        r_obj = note.Rest()
        r_obj.quarterLength = dur
        return r_obj

    else:
        return None

def build_music21_stream(token_sequence):
    """
    Converts a list of tokens into a music21 stream.
    The first 3 tokens are assumed to be tempo, key, and time signature.
    """
    s = stream.Stream()

    if len(token_sequence) >= 3:
        tempo_token = token_sequence[0]
        key_token = token_sequence[1]
        time_token = token_sequence[2]

        # Tempo
        m = re.match(r"TEMPO_(\d+(\.\d+)?)", tempo_token)
        if m:
            t = tempo.MetronomeMark(number=float(m.group(1)))
            s.insert(0, t)

        # Key
        m = re.match(r"KEY_(.+)", key_token)
        if m:
            k_str = m.group(1).replace("_", " ").strip()
            try:
                k_obj = key.Key(k_str)
                s.insert(0, k_obj)
            except:
                # Attempt to fix the key string if " sharp" is present
                k_str_fixed = k_str.replace(" sharp", "#")
                try:
                    k_obj = key.Key(k_str_fixed)
                    s.insert(0, k_obj)
                except Exception as e2:
                    print("Error processing key:", k_str, e2)

        # Time Signature
        m = re.match(r"TIME_(.+)", time_token)
        if m:
            ts_str = m.group(1).replace("_", "/")
            try:
                ts_obj = meter.TimeSignature(ts_str)
                s.insert(0, ts_obj)
            except Exception as e:
                print("Error processing time signature:", e)

        events = token_sequence[3:]
    else:
        events = token_sequence

    for token in events:
        m21_event = token_to_music21(token)
        if m21_event is not None:
            s.append(m21_event)

    return s

def main():
    # Load models and mappings
    generator, discriminator, note_to_int, int_to_note, n_vocab, sequence_length = load_models_and_mappings()

    # Load training inputs from requiredFiles
    with open('requiredFiles/training_target.pkl', 'rb') as f:
        train_inputs = pickle.load(f)
    print("Training inputs shape:", train_inputs.shape)

    # Randomly select a seed sequence from the training inputs
    sample_idx = np.random.randint(0, train_inputs.shape[0])
    seed_sequence = train_inputs[sample_idx].tolist()

    # Generate a new music sequence
    target_length = 500
    generated_sequence = generate_sequence(generator, seed_sequence, target_length, sequence_length, n_vocab, temperature=1.0)
    generated_tokens = [int_to_note[token] for token in generated_sequence]

    print("Generated sequence length:", len(generated_tokens))
    print("First 50 tokens:", generated_tokens[:50])

    # Build a music21 stream from the generated tokens
    music_stream = build_music21_stream(generated_tokens)
    
    # Ensure the output directory exists
    output_dir = 'static/audio/mix'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Function to get the next available file number based on existing files in the directory
    def get_next_file_number(directory, base_name, extension):
        existing = [f for f in os.listdir(directory) if f.startswith(base_name) and f.endswith(extension)]
        max_num = 0
        for f in existing:
            num_str = f[len(base_name):-len(extension)]
            if num_str.isdigit():
                num = int(num_str)
                if num > max_num:
                    max_num = num
        return max_num + 1
    
    file_num = get_next_file_number(output_dir, "generated_music", ".mid")
    midi_file = os.path.join(output_dir, f"generated_music{file_num}.mid")
    wav_file = os.path.join(output_dir, f"generated_music{file_num}.wav")
    mp3_file = os.path.join(output_dir, f"generated_music{file_num}.mp3")
    
    # Save the music stream as a MIDI file
    music_stream.write('midi', fp=midi_file)
    print("✅ MIDI file generated successfully:", midi_file)
    
    # Convert the MIDI file to WAV using midi2audio (FluidSynth)
    from midi2audio import FluidSynth
    sound_font_path = 'requiredFiles/FluidR3_GM.sf2'
    fs = FluidSynth(sound_font=sound_font_path)
    fs.midi_to_audio(midi_file, wav_file)
    print("✅ WAV file generated successfully:", wav_file)
    
    # Convert the WAV file to MP3 using pydub
    from pydub import AudioSegment
    sound = AudioSegment.from_wav(wav_file)
    sound.export(mp3_file, format="mp3")
    print("✅ MP3 file generated successfully:", mp3_file)

if __name__ == "__main__":
    main()
