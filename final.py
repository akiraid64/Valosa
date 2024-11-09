import torch
import numpy as np
import os
from transformers import pipeline
from diffusers.utils import load_image

# Define the path to the image folder
image_folder = "/content/top_images"

# Input the filename of the image you want to use
selected_image_name = input("Enter the exact name of the image file (e.g., 'top_image_1.jpg'): ")
selected_image_path = os.path.join(image_folder, selected_image_name)

# Check if the selected image exists
if not os.path.isfile(selected_image_path):
    raise FileNotFoundError(f"The image '{selected_image_name}' was not found in '{image_folder}'. Please check the filename.")
else:
    print(f"Selected image: {selected_image_path}")

# Load the selected image
image = load_image(selected_image_path)

# Function to get the depth map
def get_depth_map(image, depth_estimator):
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]  # Add a channel dimension
    image = np.concatenate([image, image, image], axis=2)  # Repeat to make it 3-channel
    detected_map = torch.from_numpy(image).float() / 255.0
    depth_map = detected_map.permute(2, 0, 1)  # Change to (C, H, W)
    return depth_map

# Initialize the depth estimation pipeline
depth_estimator = pipeline("depth-estimation")

# Get the depth map of the selected image
depth_map = get_depth_map(image, depth_estimator).unsqueeze(0).half()

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
depth_map = depth_map.to(device)

print("Depth map successfully generated and moved to device:", device)

from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
from diffusers.utils import make_image_grid

output = pipe(
    "make it into an cyberpunk performer show the whole face and make him cool and hot with a cool hairstyle and a fashionable sense of dressing", image=image, control_image=depth_map,
).images[0]

make_image_grid([image, output], rows=1, cols=2)
