import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips

def process_video(input_file, output_file, gradient_color = (128, 113, 37), row_percent = 0.25, log_scale = 5):
    # Open the input video
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create VideoWriter object
    temp_video = 'temp_video.mp4'
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    
    # Define the gradient
    gradient_color = np.array(gradient_color, dtype=np.uint8)
    
    # Calculate the gradient rows
    gradient_height = int(row_percent * height)
    
    # Create gradient masks
    top_gradient = np.ones((width, 3), dtype=np.uint8)*gradient_color
    bottom_gradient = np.ones((width, 3), dtype=np.uint8)*gradient_color
    
    alphas = []  
    for i in range(gradient_height):
        # Calculate logarithmic alpha
        alphas.append(np.power(np.log1p(i) / np.log1p(gradient_height - 1), log_scale))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame vertically
        flipped_frame = cv2.flip(frame, 1)
        
        # Apply the gradient to the top and bottom
        for i,alpha in enumerate(alphas):
            flipped_frame[i] = cv2.addWeighted(flipped_frame[i], alpha, top_gradient, 1-alpha, 0)
            flipped_frame[-i] = cv2.addWeighted(flipped_frame[-i], alpha, bottom_gradient, 1-alpha, 0)
        
        # Write the processed frame to the output video
        out.write(flipped_frame)
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Combine video and audio
    try:
        video_clip = VideoFileClip(temp_video)
        original_clip = VideoFileClip(input_file)
        final_clip = video_clip.set_audio(original_clip.audio)
        final_clip.write_videofile(output_file, codec='libx264')
    except Exception as e:
        print(f"Error during audio processing: {e}")