import torch
from transformers import pipeline, WhisperForConditionalGeneration, WhisperTokenizer, WhisperProcessor

import google.generativeai as genai
import os
import Levenshtein
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip
import pickle
import ffmpeg

def find_best_matches(list_a, list_b):
    indices = []
    for text_b in list_b:
        min_distance = float('inf')
        best_idx = None
        for idx, text_a in enumerate(list_a):
            distance = Levenshtein.distance(text_b, text_a)
            if distance < min_distance:
                min_distance = distance
                best_idx = idx
        indices.append(best_idx)
    return indices

def chunking_video_by_transcript(chunk_path, top_k = 2, min_chunk = 30, highlight_window = 1800):
    transcript_path = chunk_path.replace('.mp4','.pkl')
    audio_mp3_path = chunk_path.replace('.mp4','.mp3')
    if not os.path.isfile(audio_mp3_path):
        audio_segment = AudioSegment.from_file(chunk_path)
        audio_segment.export(audio_mp3_path)
    else:
        audio_segment = AudioSegment.from_file(audio_mp3_path)
    if not os.path.isfile(transcript_path):
        MODEL_ID = "openai/whisper-large-v3"
        LANGUAGE = "vi"
        TASK = "transcribe"
        device = "cuda:0"
        # device = "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        tokenizer = WhisperTokenizer.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
        processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
        #
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)

        model_oai_ft_v3 = WhisperForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True, 
            use_safetensors=True,
        )
        model_oai_ft_v3 = model_oai_ft_v3.to(device)
        batch_size=30
        pipe_oai_ft_v3 = pipeline(
            "automatic-speech-recognition",
            model=model_oai_ft_v3,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=256,
            chunk_length_s=10,
            batch_size=batch_size,
            # return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
            generate_kwargs={
                "task": "transcribe",
                "language": "vi",
                "no_repeat_ngram_size": 4, # Avoid repetition
                "return_timestamps": True,
                },
        )

        generate_kwargs={
            "task": "transcribe",
            "language": "vi",
            "no_repeat_ngram_size": 4, # Avoid repetition
            "temperature": 0.666
            }
        print("Transcribe video ... ")
        transcriptions = pipe_oai_ft_v3(audio_mp3_path, generate_kwargs=generate_kwargs, return_timestamps=True )
        with open(transcript_path, 'wb') as file:
            pickle.dump(transcriptions, file)
    else:
        with open(transcript_path, 'rb') as file:
            transcriptions = pickle.load(file)

    gemini_key = os.environ.get("GEMINI_KEY")
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash-002")

    prompt_template = """Context:
    I have a transcript from a livestream video about a sale.
    The transcript is from a sale livestream.
    The goal is to highlight key moments that would attract and inform viewers.

    The livestream transcription:
    {transcription}

    Instructions:
    Analyze the transcript to determine which sentences best represent the excitement and key points of the sale.
    Select sentences that showcase important offers, exclusive deals, or engaging interactions.
    Provide a list of the top {top_k} sentences that encapsulate the livestream’s highlights.
    Please return top {top_k} sentence only, no explain, every sentence in a line """

    generation_config=genai.types.GenerationConfig(
            temperature=.666,
        )
    start_idx = 0
    list_a = list(map(lambda x:x['text'], transcriptions['chunks']))
    list_b = []
    print("Analyzing video ...")
    for end_idx in range(len(transcriptions['chunks'])):
        if transcriptions['chunks'][end_idx]['timestamp'][0] - transcriptions['chunks'][start_idx]['timestamp'][0] < highlight_window:
            continue
        transcription_text = " ".join(map(lambda x:x['text'], transcriptions['chunks'][start_idx:end_idx]))
        print(f"Analyzing {start_idx} to {end_idx} ...")
        response = model.generate_content(prompt_template.format(transcription = transcription_text, top_k = top_k), generation_config=generation_config)
        print("choosed text:")
        print(response.text)
        list_b += list(filter(lambda x:len(x) >1, response.text.split('\n')))
        start_idx = end_idx
    
    video = VideoFileClip(chunk_path)
    output_files = []
    for i, idx in enumerate(find_best_matches(list_a, list_b)):
        try:
            transcriptions['chunks'][idx]
            start = transcriptions['chunks'][idx]['timestamp'][0]
            end = transcriptions['chunks'][idx]['timestamp'][0]
            padding = abs(min_chunk-(end-start))/2
            start = (max(0,start-padding))*1000
            end = (end + padding)*1000
            chunk_audio = audio_segment[start:end]

            # Export to mp4
            file_name = chunk_path.replace('.mp4', "").replace("/mnt/disk2/tiktok_data/raw_vid","/mnt/disk2/tiktok_data/processed_vid")
            os.makedirs(f"{file_name}", exist_ok=True)
            output_audio_file = f"{file_name}/chunk_{i}.wav"
            chunk_audio.export(output_audio_file, format="wav")
            chunk_audio_clip= AudioFileClip(output_audio_file)

            # Merge with the original video
            output_video_file = f"{file_name}/chunk_llm_{i}.mp4"
            output_clip = video.subclip(start / 1000, end / 1000)
            output_clip = output_clip.set_audio(chunk_audio_clip)
            output_clip.write_videofile(output_video_file, codec="libx264", audio_codec="aac")

            output_files.append(output_video_file)

            # Clean up temporary audio files
            os.remove(output_audio_file)
        except Exception as e:
            print(e)
    return output_files
