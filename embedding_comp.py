import os
import torch
from PIL import Image
from torchvision import transforms
import clip  # CLIP model from OpenAI

# Define the preprocessing function
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for pretrained models
])

# Define the path to your frames folder in Colab
folder_path = "/content/frame"

# Function to load and preprocess images from the folder
def load_and_preprocess_images(folder_path):
    all_images = []
    image_paths = []

    for img_file in sorted(os.listdir(folder_path)):
        if img_file.endswith((".jpg", ".png")):
            img_path = os.path.join(folder_path, img_file)
            try:
                image = Image.open(img_path).convert("RGB")
                preprocessed_image = preprocess(image)
                all_images.append(preprocessed_image)
                image_paths.append(img_path)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

    return all_images, image_paths

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the CLIP model
model, preprocess_clip = clip.load("ViT-B/32", device=device)  # or you can use "RN50" for ResNet50-based CLIP

# Load and preprocess images
all_images, image_paths = load_and_preprocess_images(folder_path)

# Perform batch processing to get image embeddings
if all_images:
    input_images = torch.stack(all_images).to(device)  # Move to GPU/CPU based on availability

    with torch.no_grad():
        # Get image features from CLIP model
        image_embeddings = model.encode_image(input_images)

    print("Image embeddings extracted successfully.")
else:
    print("No images found in the folder.")

query = "a thumbnail for a depressed movie "  # Input Query

query_tokens = clip.tokenize([query])  # Tokenize Before Embeddings

with torch.no_grad():
  query_embeddings = model.encode_text(query_tokens)

def calculate_similarity(query_embeddings, input_embeddings):
  similariries = query_embeddings @ input_embeddings.T
  return similariries

sim = calculate_similarity(query_embeddings, image_embeddings)

sim_dict= dict(zip(range(len(sim[0])), sim[0]))  # Use Dictionary to Sort the Results
sorted_sim = sorted(sim_dict.items(),key=lambda x:x[1],reverse=True)
top_sim = sorted_sim[:6]  # Get top 3 results

# Ensure the index is within the range of all_images
for i in top_sim:
  index = i[0]
  if index < len(all_images):  # Check if index is valid
    display(all_images[index])
  else:
    print(f"Index {index} is out of range for all_images (length: {len(all_images)})")

import torch
import numpy as np
from PIL import Image
import os
from torchvision import transforms

# Function to calculate cosine similarity
def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))

# Convert similarity scores to percentage
def similarity_to_percentage(similarity_scores):
    min_score = torch.min(similarity_scores)
    max_score = torch.max(similarity_scores)
    return [(score - min_score) / (max_score - min_score) * 100 for score in similarity_scores]

# Assuming query_embeddings and image_embeddings are tensors
# Here's how to calculate similarity scores for all images against the query:

# Define the path to your frames folder in Colab
folder_path = "/content/frame"  # Folder where your images are stored

# Example query embedding (replace with your actual query embedding)
query_embedding = torch.randn(512)  # Placeholder, should be the actual query embedding tensor

# Assuming `image_embeddings` are the embeddings of all images
# Sample image embeddings (replace these with actual image embeddings)
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

# Get the top 3 most similar images
top_sim = sorted_sim[:6]

# Display the top 3 results with their similarity percentages
for image_path, similarity_percentage in top_sim:
    print(f"Image: {os.path.basename(image_path)}, Similarity: {similarity_percentage:.2f}%")
    img = Image.open(image_path)  # Open the image
    img.show()  # Show the image

