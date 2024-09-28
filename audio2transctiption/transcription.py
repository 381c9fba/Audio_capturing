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
    video = VideoFileClip(video_path)
    audio = video.subclip(start_time, end_time).audio
    audio.write_audiofile(output_audio_path)
    print(f"Extracted audio: {output_audio_path}")

# Step 2: Resample the audio to 16kHz for the model and convert to mono
def resample_audio(audio_path, target_rate=16000):
    if not os.path.exists(audio_path):
        raise OSError(f"Audio file {audio_path} not found!")
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono by averaging channels
    if sample_rate != target_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_rate)
        waveform = resampler(waveform)
    return waveform, target_rate

# Step 3: Noise reduction using noisereduce
def reduce_noise(waveform, sample_rate):
    waveform_np = waveform.squeeze().numpy()
    reduced_noise = nr.reduce_noise(y=waveform_np, sr=sample_rate)
    return torch.from_numpy(reduced_noise).unsqueeze(0)  # Add channel dimension

# Step 4: Trim silence using librosa
def trim_silence(waveform, sample_rate):
    waveform_np = waveform.squeeze().numpy()
    trimmed_waveform, _ = librosa.effects.trim(waveform_np, top_db=20)
    return torch.from_numpy(trimmed_waveform).unsqueeze(0)  # Add channel dimension

# Step 5: Normalize audio
def normalize_audio(waveform):
    return (waveform - waveform.mean()) / waveform.std()

# Step 6: Complete audio preprocessing pipeline
def preprocess_audio(audio_path, preprocessed_audio_path, target_rate=16000):
    if os.path.exists(preprocessed_audio_path):
        print(f"Preprocessed audio found: {preprocessed_audio_path}")
        waveform, sample_rate = torchaudio.load(preprocessed_audio_path)
        waveform = waveform.squeeze()
        if waveform.dim() > 1:
            waveform = torch.mean(waveform, dim=0)
        return waveform, sample_rate

    print(f"Preprocessing audio: {audio_path}")
    waveform, sample_rate = resample_audio(audio_path, target_rate)
    waveform = reduce_noise(waveform, sample_rate)
    waveform = trim_silence(waveform, sample_rate)
    waveform = normalize_audio(waveform)
    waveform = waveform.squeeze()  # Ensure waveform is 1D
    torchaudio.save(preprocessed_audio_path, waveform.unsqueeze(0), sample_rate)  # Save as [1, seq_length]
    print(f"Saved preprocessed audio: {preprocessed_audio_path}")
    return waveform, sample_rate

# Step 7: Transcribe the audio using wav2vec 2.0
def transcribe_audio(waveform, sample_rate, transcription_cache_path):
    if os.path.exists(transcription_cache_path):
        print(f"Transcription loaded from cache: {transcription_cache_path}")
        with open(transcription_cache_path, "r") as f:
            return f.read()
    print("Transcribing audio...")
    print(f"Waveform shape before processing: {waveform.shape}")
    waveform = waveform.squeeze()
    if waveform.dim() > 1:
        waveform = torch.mean(waveform, dim=0)
    waveform = waveform.view(-1)
    print(f"Waveform shape after reshaping: {waveform.shape}")
    input_values = processor(waveform, return_tensors="pt", padding="longest", sampling_rate=sample_rate).input_values
    print(f"Input values shape: {input_values.shape}")
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    with open(transcription_cache_path, "w") as f:
        f.write(transcription)
    return transcription


# Step 8: Adjust time codes if transcription starts/ends mid-word
def adjust_time_codes_if_needed(transcription, start_time, end_time):
    if transcription and transcription[0].islower():
        start_time = max(0, start_time - 0.5)
    if transcription and transcription[-1].islower():
        end_time = end_time + 0.5
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
video_path = "./audio2transctiption/test.mp4"
start_time = 10.0
end_time = 15.0
audio_output_path = "./temp_audio.wav"
preprocessed_audio_path = "./preprocessed_audio.wav"
transcription_cache_path = "./transcription.txt"

final_transcription = detect_words_in_span(video_path, start_time, end_time, audio_output_path, preprocessed_audio_path, transcription_cache_path)
print(f"Final Transcription: {final_transcription}")
