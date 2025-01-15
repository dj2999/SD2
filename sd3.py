import numpy as np
import cv2
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Load Stable Diffusion model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = StableDiffusionPipeline.from_pretrained("wavymulder/Analog-Diffusion").to(device)

# Input and output video paths
input_video_path = "input.mp4"
output_video_path = "output_upscaled.mp4"

# Open the input video
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * 2  # Upscale width by 2
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * 2  # Upscale height by 2

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_index = 0
max_frames = 20  # Stop after processing 20 frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_index >= max_frames:
        break
    
    print(f"Processing frame {frame_index + 1}...")
    
    # Convert frame (OpenCV uses BGR format) to RGB for Stable Diffusion
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # Process with Stable Diffusion
    prompt = "convert to black and white old time image"
    processed_image = pipeline(prompt, init_image=pil_image, strength=0.2, num_inference_steps=30).images[0]
    
    # Resize processed image to match scaled video dimensions
    processed_image = processed_image.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
    
    # Convert back to OpenCV format (BGR)
    processed_frame = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
    
    # Write frame to the output video
    if processed_frame.shape[:2] == (frame_height, frame_width):
        out.write(processed_frame)
    else:
        print(f"Skipping frame {frame_index} due to dimension mismatch")
    
    frame_index += 1

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Output video saved. Processed {frame_index} frames.")
