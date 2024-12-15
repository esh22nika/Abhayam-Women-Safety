import os
import cv2
import base64
import google.generativeai as genai
import numpy as np

# Configure Gemini API key and model
genai.configure(api_key="AIzaSyAkOBX6VnDfPRPjzrbBt6YSQrfaLrKerYI")
model = genai.GenerativeModel("gemini-1.5-flash")

# Function to perform image analysis using Gemini and return a concise description
def analyze_image_with_gemini(image_path):
    """
    Perform image analysis using Gemini API to get a high-level, short description.
    """
    try:
        with open(image_path, "rb") as img_file:
            image_data = img_file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Adjusted prompt to ask Gemini for a concise description
        prompt = "Analyze the image and provide a concise description indicating if the scene suggests violence, aggression, or potential harm."
        response = model.generate_content([prompt, {'mime_type': 'image/jpeg', 'data': image_base64}])
        return response.text.strip()
    except Exception as e:
        return f"Error analyzing image with Gemini: {str(e)}"

# Function to determine threat level based on Gemini's analysis
def determine_threat_level_based_on_gemini(analysis):
    """
    Dynamically assess the threat level based on Gemini's analysis text.
    """
    try:
        if any(word in analysis.lower() for word in ['fight', 'violent', 'gun', 'knife', 'attack', 'weapon', 'blood']):
            return "High"
        elif any(word in analysis.lower() for word in ['aggressive', 'suspicious', 'hostile', 'tense']):
            return "Medium"
        elif any(word in analysis.lower() for word in ['calm', 'peaceful', 'neutral', 'harmless', 'safe']):
            return "Low"
        else:
            return "Unknown"
    except Exception as e:
        print(f"Error in threat level determination: {str(e)}")
        return "Unknown"

# Function to overlay description, Gemini analysis, and threat level on the image
def overlay_description_on_image(image_path, gemini_analysis, threat_level):
    """
    Overlay the Gemini analysis and threat level onto the image.
    """
    image = cv2.imread(image_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 255)  # Red color for the text
    thickness = 2

    # Positions for overlay text
    analysis_position = (50, 50)
    threat_position = (50, 100)

    # Overlay Gemini analysis and threat level
    cv2.putText(image, f"Gemini Analysis: {gemini_analysis}", analysis_position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    cv2.putText(image, f"Threat Level: {threat_level}", threat_position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    # Save the modified image
    modified_image_path = image_path.replace(".jpg", "_modified.jpg").replace(".jpeg", "_modified.jpeg").replace(".png", "_modified.png")
    cv2.imwrite(modified_image_path, image)
    print(f"Modified image saved as {modified_image_path}")

    # Delete the original image
    os.remove(image_path)
    print(f"Deleted original image: {image_path}")

# Function to process image and determine if it's a threat
def process_image(image_path):
    """
    Process the image, analyze it using Gemini, and take action based on the analysis.
    """
    try:
        # Skip already modified images
        if "_modified" in image_path:
            print(f"Skipping already modified image: {image_path}")
            return
        
        # Step 1: Analyze image with Gemini
        gemini_analysis = analyze_image_with_gemini(image_path)

        # Step 2: Determine threat level based on Gemini analysis
        threat_level = determine_threat_level_based_on_gemini(gemini_analysis)

        # If no threat is detected, delete the image
        if threat_level == "Low" or threat_level == "Unknown":
            os.remove(image_path)
            print(f"No threat detected. Deleted image: {image_path}")
            return
        
        # Step 3: Overlay analysis and threat level on the image
        overlay_description_on_image(image_path, gemini_analysis, threat_level)
             
    except Exception as e:
        print(f"Error processing image: {str(e)}")

# Function to recursively search for folders named 'violence_against_women'
def find_violence_folders(root_directory):
    """
    Recursively search for all subdirectories named 'violence_against_women'.
    """
    for root, dirs, files in os.walk(root_directory):
        if 'violence_against_women' in dirs:  # Check if folder name matches
            violence_folder_path = os.path.join(root, 'violence_against_women')
            print(f"Processing folder: {violence_folder_path}")
            for file in os.listdir(violence_folder_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(violence_folder_path, file)
                    process_image(image_path)

# Main function to process the entire directory
def main():
    directory_path = "./"  # Root directory path, change this if needed
    find_violence_folders(directory_path)

if __name__ == "__main__":
    main()
