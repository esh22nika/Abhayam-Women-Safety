import torch
import cv2
import os
from PIL import Image
from clip import clip
import numpy as np

# Load YOLOv5 model for detecting people
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use small model for fast inference

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device)

# Function to detect people in an image using YOLOv5
# Function to detect people in an image using YOLOv5
def detect_people(image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Perform inference with YOLO
    results = yolo_model(img)
    
    # Count the number of people (class 0 is for 'person')
    person_count = 0
    for result in results.xyxy[0].cpu().numpy():  # Move tensor to CPU before converting to NumPy
        if result[5] == 0:  # class_id for person
            person_count += 1
    
    return person_count

# Function to use CLIP to predict gender based on text prompts
def classify_gender(image_path):
    # Open the image and preprocess for CLIP
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Define text prompts for gender classification
    text_prompts = ["a photo of a woman", "a photo of a man"]
    text_inputs = clip.tokenize(text_prompts).to(device)

    # Get image and text embeddings from the model
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_inputs)

    # Calculate similarity between image and text features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (image_features @ text_features.T).squeeze(0)  # cosine similarity between image and text

    return similarity[0].item(), similarity[1].item()  # (woman_score, man_score)

# Function to delete images with more than 1 female detected
def delete_images_with_more_than_one_female(folder_path):
    # List all image files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it's an image file (jpg, jpeg, png)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing {filename}...")
            
            # Detect people in the image
            person_count = detect_people(file_path)
            female_count = 0

            # If there are people in the image, classify each person with CLIP
            if person_count > 0:
                for _ in range(person_count):
                    woman_score, man_score = classify_gender(file_path)
                    if woman_score > man_score:
                        female_count += 1
            
            # If more than one female detected, delete the image
            if female_count > 1:
                print(f"Deleting {filename} because it has more than 1 female.")
                os.remove(file_path)
            else:
                print(f"{filename} has {female_count} female(s) and is kept.")
                

# Function to search for 'one_female' folders in the current directory and subdirectories
def process_one_female_folders_in_directory(base_directory):
    for root, dirs, files in os.walk(base_directory):
        # Check if the 'one_female' folder exists in the current directory
        if 'one_female' in dirs:
            folder_path = os.path.join(root, 'one_female')
            print(f"Found 'one_female' folder in: {folder_path}")
            delete_images_with_more_than_one_female(folder_path)

# Main function to start the process
if __name__ == "__main__":
    # Get the current working directory
    base_directory = os.getcwd()  # Or specify the path if it's different
    process_one_female_folders_in_directory(base_directory)
