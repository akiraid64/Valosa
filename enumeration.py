import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms

# Function to calculate cosine similarity
def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))

# Convert similarity scores to percentage
def similarity_to_percentage(similarity_scores):
    min_score = torch.min(similarity_scores)
    max_score = torch.max(similarity_scores)
    return [(score - min_score) / (max_score - min_score) * 100 for score in similarity_scores]

# Define the path to your frames folder in Colab
folder_path = "/content/frame"  # Change this to your frames folder path

# Example query embedding (replace with your actual query embedding)
query_embedding = torch.randn(512)  # Placeholder, should be the actual query embedding tensor

# Assuming `image_embeddings` are the embeddings of all images
image_embeddings = torch.randn(100, 512)  # 100 images, each having 512 dimensions

# Calculate similarity between query and all image embeddings
similarities = [cosine_similarity(query_embedding, img_embedding) for img_embedding in image_embeddings]

# Convert similarity scores to percentages
sim_percentages = similarity_to_percentage(torch.tensor(similarities))

# Load image paths from the folder
image_paths = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.endswith(('.jpg', '.png'))]

# Create a dictionary of image paths and similarity percentages
sim_dict = dict(zip(image_paths, sim_percentages))

# Sort the dictionary by similarity percentage in descending order
sorted_sim = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)

# Get the top 5 most similar images
top_sim = sorted_sim[:6]

# Save each of the top 5 images as separate files
output_folder = "/content/top_images"
os.makedirs(output_folder, exist_ok=True)

for idx, (image_path, _) in enumerate(top_sim):
    img = Image.open(image_path)
    output_path = os.path.join(output_folder, f"top_image_{idx + 1}.jpg")
    img.save(output_path)
    print(f"Saved: {output_path}")
