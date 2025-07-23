import whisper

def transcribe_with_word_timestamps(audio_path, model_size="medium"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, word_timestamps=True, verbose=False)

    word_times = []
    for segment in result["segments"]:
        for word_info in segment["words"]:
            word = word_info["word"].strip()
            start = word_info["start"]
            end = word_info["end"]
            word_times.append((word, start, end))

    return word_times

# Example usage:
timestamps = transcribe_with_word_timestamps("output_ar_inpaint.wav")
for word, start, end in timestamps:
    print(f"{word:15} {start:.2f} sec → {end:.2f} sec")

def seconds_to_ar_token_range(start_time, end_time, num_tokens, audio_duration):
    """
    Map a (start_time, end_time) to AR token indices.
    
    Args:
        start_time (float): start of word/phrase in seconds
        end_time (float): end of word/phrase in seconds
        num_tokens (int): total number of AR tokens
        audio_duration (float): total duration of the generated audio (in seconds)
        
    Returns:
        tuple of (start_index, end_index) for AR token slicing
    """
    tokens_per_sec = num_tokens / audio_duration
    start_idx = int(start_time * tokens_per_sec)
    end_idx = int(end_time * tokens_per_sec)
    return start_idx, end_idx