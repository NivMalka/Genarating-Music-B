import os
import re
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
import music21
from music21 import stream, note, chord, tempo, key, meter

# ייבוא המחלקות מהקובץ models.py
from models import SmallMusicGenerator, SmallMusicDiscriminator

# הגדרת המכשיר (GPU אם קיים)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models_and_mappings():
    """
    טוען את המשקלים של המודל ואת קבצי המיפויים (note_to_int ו-int_to_note) 
    מתוך התיקייה requiredFiles, ומחזיר את המודלים והנתונים הדרושים.
    """

    # טען את המיפויים
    with open('requiredFiles/note_to_int.pkl', 'rb') as f:
        note_to_int = pickle.load(f)
    with open('requiredFiles/int_to_note.pkl', 'rb') as f:
        int_to_note = pickle.load(f)

    n_vocab = len(note_to_int)
    sequence_length = 200  # ניתן לשנות לפי מה שהוגדר באימון

    # אתחול המודלים
    generator = SmallMusicGenerator(n_vocab=n_vocab, sequence_length=sequence_length)
    discriminator = SmallMusicDiscriminator(n_vocab=n_vocab, sequence_length=sequence_length)

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    generator.positional_encoding = generator.positional_encoding.to(device)

    # טען משקלים
    checkpoint_path = "requiredFiles/best_model_Gloss_0.0319.pth"  # עדכן לפי שם הקובץ אם שונה
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    # מצב הערכה
    generator.eval()
    discriminator.eval()

    print("✅ Models and mappings loaded successfully.")
    return generator, discriminator, note_to_int, int_to_note, n_vocab, sequence_length

def generate_sequence(generator, seed_sequence, target_length, sequence_length, n_vocab, temperature=1.0):
    """
    מייצר רצף מוזיקלי ארוך באמצעות הגנרטור:
    - seed_sequence: רשימת אינדקסים התחלתית
    - target_length: אורך סופי רצוי של הרצף (מספר טוקנים)
    - temperature: משפיע על פיזור ההסתברויות (ככל שגבוה יותר, התוצאה מגוונת יותר)
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

# מיפוי pitch class ל-Note Names
pitch_class_map = {
    0: "C", 1: "C#", 2: "D", 3: "D#",
    4: "E", 5: "F", 6: "F#", 7: "G",
    8: "G#", 9: "A", 10: "A#", 11: "B"
}

def parse_duration(dur_str, scale=1.0):
    """
    ממיר משך ביחידות 16th ל-quarterLength של music21
    """
    try:
        dur = float(dur_str)
    except:
        dur = 1.0
    return (dur / 4.0) * scale

def token_to_music21(token):
    """
    ממיר טוקן (NOTE_/CHORD_/REST_) לאובייקט מתאים של music21.
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
    ממיר רשימת טוקנים לזרם music21, כאשר 3 הטוקנים הראשונים משמשים כטמפו, מפתח וחתימת זמן.
    """
    s = stream.Stream()

    if len(token_sequence) >= 3:
        tempo_token = token_sequence[0]
        key_token = token_sequence[1]
        time_token = token_sequence[2]

        # טמפו
        m = re.match(r"TEMPO_(\d+(\.\d+)?)", tempo_token)
        if m:
            t = tempo.MetronomeMark(number=float(m.group(1)))
            s.insert(0, t)

        # מפתח
        m = re.match(r"KEY_(.+)", key_token)
        if m:
            k_str = m.group(1).replace("_", " ").strip()
            try:
                k_obj = key.Key(k_str)
                s.insert(0, k_obj)
            except:
                # ניסיון תיקון במקרה של " sharp"
                k_str_fixed = k_str.replace(" sharp", "#")
                try:
                    k_obj = key.Key(k_str_fixed)
                    s.insert(0, k_obj)
                except Exception as e2:
                    print("Error processing key:", k_str, e2)

        # חתימת זמן
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
    # טוען את המודלים והמיפויים
    generator, discriminator, note_to_int, int_to_note, n_vocab, sequence_length = load_models_and_mappings()

    # טוען מערך האימון (לבחירת seed) מתוך התיקייה requiredFiles
    with open('requiredFiles/training_target.pkl', 'rb') as f:
        train_inputs = pickle.load(f)
    print("Training inputs shape:", train_inputs.shape)

    # בוחר seed אקראי
    sample_idx = np.random.randint(0, train_inputs.shape[0])
    seed_sequence = train_inputs[sample_idx].tolist()

    # יוצר רצף ארוך יותר
    target_length = 500
    generated_sequence = generate_sequence(generator, seed_sequence, target_length, sequence_length, n_vocab, temperature=1.0)
    generated_tokens = [int_to_note[token] for token in generated_sequence]

    print("Generated sequence length:", len(generated_tokens))
    print("First 50 tokens:", generated_tokens[:50])

    # בניית זרם מוזיקלי וייצוא ל-MIDI
    music_stream = build_music21_stream(generated_tokens)

    # נתיב הקובץ יישמר בתיקייה static/audio/mix
    output_midi_path = 'static/audio/mix/generated_music.mid'
    music_stream.write('midi', fp=output_midi_path)
    print("✅ MIDI file generated successfully:", output_midi_path)

    # המרה מ-MIDI ל-WAV בעזרת midi2audio (FluidSynth)
    from midi2audio import FluidSynth

    midi_file = output_midi_path  # 'static/audio/mix/generated_music.mid'
    wav_file = 'static/audio/mix/generated_music.wav'
    sound_font_path = 'requiredFiles/FluidR3_GM.sf2'

    fs = FluidSynth(sound_font=sound_font_path)
    fs.midi_to_audio(midi_file, wav_file)
    print("✅ WAV file generated successfully:", wav_file)

    # המרה מ-WAV ל-MP3 בעזרת pydub
    from pydub import AudioSegment

    sound = AudioSegment.from_wav(wav_file)
    mp3_file = 'static/audio/mix/generated_music.mp3'
    sound.export(mp3_file, format="mp3")
    print("✅ MP3 file generated successfully:", mp3_file)


if __name__ == "__main__":
    main()
