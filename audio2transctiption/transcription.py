import os
from moviepy.editor import VideoFileClip
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import noisereduce as nr
import librosa

# Load the wav2vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")

# Step 1: Extract audio from video between two timecodes
def extract_audio(video_path, start_time, end_time, output_audio_path):
    if os.path.exists(output_audio_path):
        print(f"Audio already extracted: {output_audio_path}")
        return
    if not os.path.exists(video_path):
        raise OSError(f"MoviePy error: the file {video_path} could not be found! Please check the path.")
    try:
        video = VideoFileClip(video_path)
        audio = video.subclip(start_time, end_time).audio
        if audio is None:
            raise ValueError(f"No audio stream found in {video_path}.")
        audio.write_audiofile(output_audio_path, logger=None)
        print(f"Extracted audio: {output_audio_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract audio: {e}")

# Step 2: Resample the audio to 16kHz for the model and convert to mono
def resample_audio(audio_path, target_rate=16000):
    if not os.path.exists(audio_path):
        raise OSError(f"Audio file {audio_path} not found!")
    waveform, sample_rate = torchaudio.load(audio_path)
    print(f"Original waveform shape: {waveform.shape}")
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono by averaging channels
        print(f"Converted to mono, new waveform shape: {waveform.shape}")
    if sample_rate != target_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_rate)
        waveform = resampler(waveform)
        print(f"Resampled waveform shape: {waveform.shape}")
    return waveform, target_rate

# Step 3: Noise reduction using noisereduce
def reduce_noise(waveform, sample_rate):
    waveform_np = waveform.squeeze().numpy()
    reduced_noise = nr.reduce_noise(y=waveform_np, sr=sample_rate)
    reduced_noise_tensor = torch.from_numpy(reduced_noise).unsqueeze(0)  # Add channel dimension
    print(f"After noise reduction, waveform shape: {reduced_noise_tensor.shape}")
    return reduced_noise_tensor

# Step 4: Trim silence using librosa
def trim_silence(waveform, sample_rate):
    waveform_np = waveform.squeeze().numpy()
    trimmed_waveform, _ = librosa.effects.trim(waveform_np, top_db=20)
    trimmed_waveform_tensor = torch.from_numpy(trimmed_waveform).unsqueeze(0)  # Add channel dimension
    print(f"After trimming silence, waveform shape: {trimmed_waveform_tensor.shape}")
    return trimmed_waveform_tensor

# Step 5: Normalize audio
def normalize_audio(waveform):
    normalized_waveform = (waveform - waveform.mean()) / waveform.std()
    print(f"After normalization, waveform mean: {normalized_waveform.mean().item()}, std: {normalized_waveform.std().item()}")
    return normalized_waveform

# Step 6: Complete audio preprocessing pipeline
def preprocess_audio(audio_path, preprocessed_audio_path, target_rate=16000):
    if os.path.exists(preprocessed_audio_path):
        print(f"Preprocessed audio found: {preprocessed_audio_path}")
        waveform, sample_rate = torchaudio.load(preprocessed_audio_path)
        print(f"Loaded preprocessed waveform shape: {waveform.shape}")
        waveform = waveform.squeeze()
        if waveform.dim() > 1:
            waveform = torch.mean(waveform, dim=0)
            print(f"Converted to mono during load, waveform shape: {waveform.shape}")
        return waveform, sample_rate

    print(f"Preprocessing audio: {audio_path}")
    waveform, sample_rate = resample_audio(audio_path, target_rate)
    waveform = reduce_noise(waveform, sample_rate)
    waveform = trim_silence(waveform, sample_rate)
    waveform = normalize_audio(waveform)
    waveform = waveform.squeeze()  # Ensure waveform is 1D
    print(f"Final waveform shape before saving: {waveform.shape}")
    torchaudio.save(preprocessed_audio_path, waveform.unsqueeze(0), sample_rate)  # Save as [1, seq_length]
    print(f"Saved preprocessed audio: {preprocessed_audio_path}")
    return waveform, sample_rate

# Step 7: Transcribe the audio using wav2vec 2.0
def transcribe_audio(waveform, sample_rate, transcription_cache_path):
    if os.path.exists(transcription_cache_path):
        print(f"Transcription loaded from cache: {transcription_cache_path}")
        with open(transcription_cache_path, "r", encoding="utf-8") as f:
            return f.read()
    print("Transcribing audio...")

    # Ensure waveform is 1D
    if waveform.dim() > 1:
        waveform = torch.mean(waveform, dim=0)
        print(f"Converted to mono during transcription, waveform shape: {waveform.shape}")

    # Ensure it's a 1D tensor
    waveform = waveform.view(-1)
    print(f"Waveform shape before processing: {waveform.shape}")

    input_values = processor(waveform, return_tensors="pt", padding="longest", sampling_rate=sample_rate).input_values
    print(f"Input values shape: {input_values.shape}")

    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    with open(transcription_cache_path, "w", encoding="utf-8") as f:
        f.write(transcription)

    print(f"Transcription saved to cache: {transcription_cache_path}")
    return transcription

# Step 8: Adjust time codes if transcription starts/ends mid-word
def adjust_time_codes_if_needed(transcription, start_time, end_time):
    if transcription and transcription[0].islower():
        start_time = max(0, start_time - 0.5)
        print(f"Adjusted start_time to: {start_time}")
    if transcription and transcription[-1].islower():
        end_time = end_time + 0.5
        print(f"Adjusted end_time to: {end_time}")
    return start_time, end_time

# Step 9: Full pipeline: extract, clean, transcribe, and adjust timecodes
def detect_words_in_span(video_path, start_time, end_time, audio_output_path, preprocessed_audio_path, transcription_cache_path):
    extract_audio(video_path, start_time, end_time, audio_output_path)
    waveform, sample_rate = preprocess_audio(audio_output_path, preprocessed_audio_path)
    transcription = transcribe_audio(waveform, sample_rate, transcription_cache_path)
    adjusted_start, adjusted_end = adjust_time_codes_if_needed(transcription, start_time, end_time)
    if (adjusted_start != start_time or adjusted_end != end_time):
        extract_audio(video_path, adjusted_start, adjusted_end, audio_output_path)
        waveform, sample_rate = preprocess_audio(audio_output_path, preprocessed_audio_path)
        transcription = transcribe_audio(waveform, sample_rate, transcription_cache_path)
    return transcription

# Example usage
def main():
    video_path = "./audio2transctiption/test.mp4"
    start_time = 10.0
    end_time = 15.0
    audio_output_path = "./temp_audio.wav"
    preprocessed_audio_path = "./preprocessed_audio.wav"
    transcription_cache_path = "./transcription.txt"

    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        return

    try:
        final_transcription = detect_words_in_span(video_path, start_time, end_time, audio_output_path, preprocessed_audio_path, transcription_cache_path)
        print(f"Final Transcription: {final_transcription}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
