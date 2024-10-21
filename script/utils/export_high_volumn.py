import argparse
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip
import tempfile
import os
import ffmpeg

def get_convert_video2mp4(input_file):
    # Create a temporary file for the MP4 output
    output_file = input_file.replace('_flv.mp4', '.mp4')
    if not os.path.isfile(output_file):
        ffmpeg.input(input_file).output(output_file, y='-y').run(quiet=False)

    return VideoFileClip(output_file)

def process_video(input_file, top_percent = 0.1, padding_start = 10000, padding_end = 10000):
    # Extract audio from video
    video = get_convert_video2mp4(input_file)
    audio = video.audio
    audio_file = "/tmp/tiktok/temp_audio.wav"
    audio.write_audiofile(audio_file, codec='pcm_s16le')

    # Load audio with pydub
    audio_segment = AudioSegment.from_file(audio_file)

    # Chunk audio into 5-second segments
    chunk_length = 5000  # 5 seconds in milliseconds
    chunks = [audio_segment[i:i + chunk_length] for i in range(0, len(audio_segment), chunk_length)]

    # Calculate dBFS for each chunk and select top 10%
    loudness = [(chunk.dBFS, i) for i, chunk in enumerate(chunks)]
    loudness.sort(reverse=True)
    top_10_percent = int(top_percent * len(loudness))
    top_chunks_indices = [index for _, index in loudness[:top_10_percent]]

    # Merge chunks that are within 10 seconds of each other
    merged_chunks = []
    current_chunk = None
    for index in sorted(top_chunks_indices):
        if current_chunk is None:
            current_chunk = [index]
        elif index - current_chunk[-1] <= 2:  # 2 chunks = 10s
            current_chunk.append(index)
        else:
            merged_chunks.append(current_chunk)
            current_chunk = [index]
    if current_chunk:
        merged_chunks.append(current_chunk)

    # Pad and export each merged chunk
    output_files = []
    for i, indices in enumerate(merged_chunks):
        start = max(0, indices[0] * chunk_length - padding_start)
        end = (indices[-1] + 1) * chunk_length + padding_end
        chunk_audio = audio_segment[start:end]

        # Export to mp4
        file_name = input_file.replace('.mp4', "").replace("/mnt/disk2/tiktok_data/raw_vid","/mnt/disk2/tiktok_data/processed_vid")
        os.makedirs(f"{file_name}", exist_ok=True)
        output_audio_file = f"{file_name}/chunk_{i}.wav"
        chunk_audio.export(output_audio_file, format="wav")
        chunk_audio_clip= AudioFileClip(output_audio_file)

        # Merge with the original video
        output_video_file = f"{file_name}/chunk_{i}.mp4"
        output_clip = video.subclip(start / 1000, end / 1000)
        output_clip = output_clip.set_audio(chunk_audio_clip)
        output_clip.write_videofile(output_video_file, codec="libx264", audio_codec="aac")

        output_files.append(output_video_file)

        # Clean up temporary audio files
        os.remove(output_audio_file)

    # Clean up
    os.remove(audio_file)

    return output_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video file.")
    parser.add_argument("input_video_file", help="The path to the input video file.")

    args = parser.parse_args()
    output_files = process_video(args.input_video_file)
    print("Exported video chunks:", output_files)