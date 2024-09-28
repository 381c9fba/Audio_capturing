import os
from moviepy.editor import VideoFileClip
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import noisereduce as nr
import librosa
import whisper  # Ensure OpenAI's Whisper is installed and there are no naming conflicts

# Initialize Wav2Vec2 processor and model
w2v_processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
w2v_model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")

# Initialize Whisper model
whisper_model = whisper.load_model("base")  # Options: 'tiny', 'base', 'small', 'medium', 'large'

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

# Step 2: Resample the audio to 16kHz for Wav2Vec2 and convert to mono
def resample_audio_wav2vec(audio_path, target_rate=16000):
    if not os.path.exists(audio_path):
        raise OSError(f"Audio file {audio_path} not found!")
    waveform, sample_rate = torchaudio.load(audio_path)
    print(f"[Wav2Vec2] Original waveform shape: {waveform.shape}")
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono by averaging channels
        print(f"[Wav2Vec2] Converted to mono, new waveform shape: {waveform.shape}")
    if sample_rate != target_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_rate)
        waveform = resampler(waveform)
        print(f"[Wav2Vec2] Resampled waveform shape: {waveform.shape}")
    return waveform, target_rate

# Step 3: Noise reduction using noisereduce
def reduce_noise(waveform, sample_rate):
    waveform_np = waveform.squeeze().numpy()
    reduced_noise = nr.reduce_noise(y=waveform_np, sr=sample_rate)
    reduced_noise_tensor = torch.from_numpy(reduced_noise).unsqueeze(0)  # Add channel dimension
    print(f"[Noise Reduction] After noise reduction, waveform shape: {reduced_noise_tensor.shape}")
    return reduced_noise_tensor

# Step 4: Trim silence using librosa
def trim_silence(waveform, sample_rate):
    waveform_np = waveform.squeeze().numpy()
    trimmed_waveform, _ = librosa.effects.trim(waveform_np, top_db=20)
    trimmed_waveform_tensor = torch.from_numpy(trimmed_waveform).unsqueeze(0)  # Add channel dimension
    print(f"[Silence Trimming] After trimming silence, waveform shape: {trimmed_waveform_tensor.shape}")
    return trimmed_waveform_tensor

# Step 5: Normalize audio
def normalize_audio(waveform):
    normalized_waveform = (waveform - waveform.mean()) / waveform.std()
    print(f"[Normalization] Waveform mean: {normalized_waveform.mean().item()}, std: {normalized_waveform.std().item()}")
    return normalized_waveform

# Step 6: Complete audio preprocessing pipeline for Wav2Vec2
def preprocess_audio_wav2vec(audio_path, preprocessed_audio_path, target_rate=16000):
    if os.path.exists(preprocessed_audio_path):
        print(f"[Preprocessing] Preprocessed audio found: {preprocessed_audio_path}")
        waveform, sample_rate = torchaudio.load(preprocessed_audio_path)
        waveform = waveform.squeeze()
        if waveform.dim() > 1:
            waveform = torch.mean(waveform, dim=0)
            print(f"[Preprocessing] Converted to mono during load, waveform shape: {waveform.shape}")
        return waveform, sample_rate

    print(f"[Preprocessing] Preprocessing audio: {audio_path}")
    waveform, sample_rate = resample_audio_wav2vec(audio_path, target_rate)
    waveform = reduce_noise(waveform, sample_rate)
    waveform = trim_silence(waveform, sample_rate)
    waveform = normalize_audio(waveform)
    waveform = waveform.squeeze()  # Ensure waveform is 1D
    print(f"[Preprocessing] Final waveform shape before saving: {waveform.shape}")
    torchaudio.save(preprocessed_audio_path, waveform.unsqueeze(0), sample_rate)  # Save as [1, seq_length]
    print(f"[Preprocessing] Saved preprocessed audio: {preprocessed_audio_path}")
    return waveform, sample_rate

# Step 7: Transcribe the audio using Wav2Vec2
def transcribe_audio_wav2vec(waveform, sample_rate, transcription_cache_path):
    if os.path.exists(transcription_cache_path):
        print(f"[Wav2Vec2] Transcription loaded from cache: {transcription_cache_path}")
        with open(transcription_cache_path, "r", encoding="utf-8") as f:
            return f.read()
    print("[Wav2Vec2] Transcribing audio...")

    # Ensure waveform is 1D
    if waveform.dim() > 1:
        waveform = torch.mean(waveform, dim=0)
        print(f"[Wav2Vec2] Converted to mono during transcription, waveform shape: {waveform.shape}")

    # Ensure it's a 1D tensor
    waveform = waveform.view(-1)
    print(f"[Wav2Vec2] Waveform shape before processing: {waveform.shape}")

    input_values = w2v_processor(waveform, return_tensors="pt", padding="longest", sampling_rate=sample_rate).input_values
    print(f"[Wav2Vec2] Input values shape: {input_values.shape}")

    with torch.no_grad():
        logits = w2v_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = w2v_processor.batch_decode(predicted_ids)[0]

    with open(transcription_cache_path, "w", encoding="utf-8") as f:
        f.write(transcription)

    print(f"[Wav2Vec2] Transcription saved to cache: {transcription_cache_path}")
    return transcription

# Step 8: Transcribe the audio using Whisper
def transcribe_audio_whisper(audio_path, transcription_cache_path):
    if os.path.exists(transcription_cache_path):
        print(f"[Whisper] Transcription loaded from cache: {transcription_cache_path}")
        with open(transcription_cache_path, "r", encoding="utf-8") as f:
            return f.read()
    print("[Whisper] Transcribing audio...")

    # Whisper expects a file path
    try:
        result = whisper_model.transcribe(audio_path, language="ru")
        transcription = result['text']
    except Exception as e:
        raise RuntimeError(f"Whisper transcription failed: {e}")

    with open(transcription_cache_path, "w", encoding="utf-8") as f:
        f.write(transcription)

    print(f"[Whisper] Transcription saved to cache: {transcription_cache_path}")
    return transcription

# Step 9: Adjust time codes if transcription starts/ends mid-word
def adjust_time_codes_if_needed(transcription, start_time, end_time):
    if transcription and transcription[0].islower():
        start_time = max(0, start_time - 0.5)
        print(f"[Adjustment] Adjusted start_time to: {start_time}")
    if transcription and transcription[-1].islower():
        end_time = end_time + 0.5
        print(f"[Adjustment] Adjusted end_time to: {end_time}")
    return start_time, end_time

# Step 10: Ensemble transcriptions from Wav2Vec2 and Whisper
def ensemble_transcriptions(transcription_wav2vec, transcription_whisper):
    # Simple ensemble method: choose the transcription with higher confidence or longer word count
    # Here, we'll choose the transcription with the higher word count
    if not transcription_wav2vec:
        return transcription_whisper
    if not transcription_whisper:
        return transcription_wav2vec
    count_wav2vec = len(transcription_wav2vec.split())
    count_whisper = len(transcription_whisper.split())
    if count_wav2vec > count_whisper:
        print("[Ensemble] Choosing Wav2Vec2 transcription based on word count.")
        return transcription_wav2vec
    else:
        print("[Ensemble] Choosing Whisper transcription based on word count.")
        return transcription_whisper

# Step 11: Full pipeline: extract, clean, transcribe, adjust timecodes, and ensemble
def detect_words_in_span(video_path, start_time, end_time,
                         audio_output_path, preprocessed_audio_path_w2v,
                         transcription_cache_path_w2v, transcription_cache_path_whisper):
    # Extract audio
    extract_audio(video_path, start_time, end_time, audio_output_path)

    # Preprocess audio for Wav2Vec2
    waveform_w2v, sample_rate_w2v = preprocess_audio_wav2vec(audio_output_path, preprocessed_audio_path_w2v)

    # Transcribe with Wav2Vec2
    transcription_wav2vec = transcribe_audio_wav2vec(waveform_w2v, sample_rate_w2v, transcription_cache_path_w2v)

    # Transcribe with Whisper
    transcription_whisper = transcribe_audio_whisper(audio_output_path, transcription_cache_path_whisper)

    # Ensemble transcriptions
    final_transcription = ensemble_transcriptions(transcription_wav2vec, transcription_whisper)
    print(f"[Ensemble] Final Transcription: {final_transcription}")

    # Adjust time codes based on transcription
    adjusted_start, adjusted_end = adjust_time_codes_if_needed(final_transcription, start_time, end_time)
    if (adjusted_start != start_time or adjusted_end != end_time):
        print("[Ensemble] Adjusting time codes and reprocessing...")
        final_transcription = detect_words_in_span(
            video_path, adjusted_start, adjusted_end,
            audio_output_path, preprocessed_audio_path_w2v,
            transcription_cache_path_w2v, transcription_cache_path_whisper
        )
    return final_transcription

# Example usage
def main():
    video_path = "./audio2transctiption/test.mp4"  # Update path as needed
    start_time = 10.0  # in seconds
    end_time = 15.0    # in seconds
    audio_output_path = "./temp_audio.wav"
    preprocessed_audio_path_w2v = "./preprocessed_audio_w2v.wav"
    transcription_cache_path_w2v = "./transcription_wav2vec.txt"
    transcription_cache_path_whisper = "./transcription_whisper.txt"

    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        return

    try:
        final_transcription = detect_words_in_span(
            video_path, start_time, end_time,
            audio_output_path, preprocessed_audio_path_w2v,
            transcription_cache_path_w2v, transcription_cache_path_whisper
        )
        print(f"\n=== Final Transcription ===\n{final_transcription}\n")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
