import math
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import os
from datetime import datetime
import pyautogui
import numpy as np
import cloudinary
import cloudinary.uploader
from twilio.rest import Client
import csv


class ViolenceTracker:
    def __init__(self, yolo_model_path, clip_model_name="openai/clip-vit-base-patch16", shrink_factor=0.2):
        # Initialize YOLO model
        self.yolo_model = YOLO(yolo_model_path)

        # Initialize CLIP model and processor for gender and violence classification
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Settings for gender detection
        self.gender_labels = ["a person who is male", "a person who is female"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shrink_factor = shrink_factor

    def get_centroid(self, box):
        centroid_x = (box[0] + box[2]) / 2
        centroid_y = (box[1] + box[3]) / 2
        return (centroid_x, centroid_y)

    # Calculate distance between centroids
    def calculate_centroid_distance(self, centroid1, centroid2):
        return math.sqrt((centroid2[0] - centroid1[0]) ** 2 + (centroid2[1] - centroid1[1]) ** 2)

    def detect_frame(self, frame):
        results = self.yolo_model.track(frame, persist=True, conf=0.3)[0]
        player_dict = {}
        for box in results.boxes:
            if box.id is not None:
                track_id = int(box.id.tolist()[0])
                bbox = list(map(int, box.xyxy.tolist()[0]))
                if results.names[int(box.cls.tolist()[0])] == "person":
                    player_dict[track_id] = bbox
        return player_dict

    def classify_action(self, cropped_image):
        self.actions = [
            "two people fighting", "a person walking", "a person standing",
            "a person running", "a person jumping", "a person dancing", "a person laughing",
            "a person eating", "a person drinking", "a person reading", "a person writing",
            "a person swimming", "a person cycling", "a person driving", "a person playing a sport",
            "a person crying", "a person falling", "a person shouting", "a person grabbing something",
            "a person slapping", "a person punching", "two people wrestling", "a person sleeping",
            "a person looking around", "a person waving", "a person running towards someone", 
            "a person kicking", "a person hitting", "a person sneaking", "a person hiding", 
            "a person climbing", "a person punching a wall", "a person screaming", 
            "a person helping someone", "a person defending themselves", "a person being aggressive", 
            "a person being attacked", "a person fleeing", "a person pulling something", 
            "a person pushing something", "a person hugging", "a person shaking hands", 
            "a person throwing an object", "a person catching an object", "a person lifting weights", 
            "a person exercising", "a person playing with a pet", "a person playing video games", 
            "a person singing", "a person posing"
        ]
        inputs = self.clip_processor(images=cropped_image, return_tensors="pt").to(self.device)
        text_inputs = self.clip_processor(text=self.actions, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            text_features = self.clip_model.get_text_features(**text_inputs)
            similarities = (image_features @ text_features.T).softmax(dim=-1)
            predicted_action_idx = similarities.argmax().item()
            return self.actions[predicted_action_idx]

    def classify_gender(self, cropped_image):
        # Gender classification using the CLIP model
        gender_inputs = self.clip_processor(images=cropped_image, return_tensors="pt").to(self.device)
        gender_text_inputs = self.clip_processor(text=self.gender_labels, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**gender_inputs)
            text_features = self.clip_model.get_text_features(**gender_text_inputs)
            similarities = (image_features @ text_features.T).softmax(dim=-1)
            predicted_gender_idx = similarities.argmax().item()
            return self.gender_labels[predicted_gender_idx]

    def detect_violence_against_women(self, player_detections, player_genders, player_labels):
        violence_against_women = False
        fighting_detected = False

        for id1, bbox1 in player_detections.items():
            for id2, bbox2 in player_detections.items():
                if id1 >= id2:
                    continue

                # Check if one person is female and the other is male
                female_id = male_id = None
                if player_genders.get(id1) == "a person who is female" and player_genders.get(id2) == "a person who is male":
                    female_id, male_id = id1, id2
                elif player_genders.get(id1) == "a person who is male" and player_genders.get(id2) == "a person who is female":
                    female_id, male_id = id2, id1

                # If a male-female pair exists, check for violent actions
                if female_id and male_id:
                    female_action = player_labels.get(female_id, "")
                    male_action = player_labels.get(male_id, "")

                    # Specific violence indicators involving male aggression towards female
                    violent_actions = [
                        "two people fighting", "a person hitting", "a person slapping", "a person punching",
                        "a person kicking", "a person grabbing something",
                        "a person being aggressive", "a person being attacked",
                        "a person defending themselves"
                    ]

                    if any(action in male_action for action in violent_actions):
                        violence_against_women = True
                        break

        # Check if "two people fighting" is detected with at least one female
        for track_id, action in player_labels.items():
            if action == "two people fighting" and "a person who is female" in player_genders.values():
                fighting_detected = True

        return violence_against_women or fighting_detected


    def classify_players(self, frame, player_detections):
        player_labels = {}
        player_genders = {}
        male_count = 0
        female_count = 0
        
        for track_id, bbox in player_detections.items():
            x1, y1, x2, y2 = map(int, bbox)
            cropped_image = frame[y1:y2, x1:x2]
            if cropped_image.size > 0:
                action = self.classify_action(cropped_image)
                gender = self.classify_gender(cropped_image)
                player_labels[track_id] = action
                player_genders[track_id] = gender
                
                # Update gender counts
                if gender == "a person who is male":
                    male_count += 1
                elif gender == "a person who is female":
                    female_count += 1
        
        # Detect violence specifically against women
        violence_against_women = self.detect_violence_against_women(player_detections, player_genders, player_labels)
        
        return player_labels, player_genders, violence_against_women, male_count, female_count

    def draw_bboxes(self, frame, player_detections, violent_pairs, player_labels, player_genders, fighting_detected, male_count, female_count):
        green_box_detected = False
        violence_detected = False
        
        for track_id, bbox in player_detections.items():
            x1, y1, x2, y2 = bbox
            color = (0, 0, 255)  # Default color (Red)
            if any(track_id in pair for pair in violent_pairs):
                color = (0, 255, 0)  # Green color for violence
                green_box_detected = True
            label = f"Player ID: {track_id} - {player_genders.get(track_id, 'Unknown')}"
            if track_id in player_labels:
                label += f" ({player_labels[track_id]})"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Add male and female counts to the frame
        cv2.putText(frame, f"Males: {male_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Females: {female_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if green_box_detected:
            violence_detected = True

        return frame, violence_detected, green_box_detected, male_count, female_count
    
    # Updated log_violence_to_csv method in violence_tracker.py
    def log_violence_to_csv(self, timestamp, action_detected, male_count, female_count, location):
        log_file = 'violence_log.csv'
        file_exists = os.path.isfile(log_file)

        with open(log_file, 'a', newline='') as csvfile:
            fieldnames = ['Timestamp', 'Action Detected', 'Male Count', 'Female Count', 'Location']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()  # Write the header only if the file doesn't exist

            writer.writerow({
                'Timestamp': timestamp,
                'Action Detected': action_detected,
                'Male Count': male_count,
                'Female Count': female_count,
                'Location': location  # Add location to the log
            })



    def save_screenshot(self, frame, violence_against_women, female_count, location):
        if violence_against_women or female_count == 1:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # Corrected the use of f-string with proper formatting
            folder = f"{location}/violence_against_women" if violence_against_women else f"{location}/one_female"
            
            os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't exist
            local_path = os.path.join(folder, f"{timestamp}.png")
            
            # Save the screenshot locally
            cv2.imwrite(local_path, frame)

            # Cloudinary configuration
            cloudinary.config(
                cloud_name="dbaacvp3e",
                api_key="714437137622323",
                api_secret="BLVWgl2D8b7b5Wjt_pMLoe5HdP4"
            )
            
            try:
                # Upload image to Cloudinary
                upload_result = cloudinary.uploader.upload(local_path)
                image_url = upload_result.get("secure_url")
                
                # Call method to send alert with image URL
                self.send_whatsapp_alert(violence_against_women, female_count, image_url)
            except Exception as e:
                # Handle exceptions during upload
                print(f"Error uploading to Cloudinary: {e}")

    def send_whatsapp_alert(self, violence_against_women, female_count, image_url):
        # Twilio configuration
        account_sid = "ACbe34242a5c03284f98c19d58bc38bb53"
        auth_token = "ecde757c677833f276cbf696af067bad"
        client = Client(account_sid, auth_token)

        from_whatsapp_number = "whatsapp:+14155238886"
        to_whatsapp_number = "whatsapp:+919372063867"

        # Set appropriate message based on the detected condition
        if violence_against_women:
            message_body = "Alert! Violence against a woman detected. See the attached image."
        elif female_count == 1 and violence_against_women == False:
            message_body = "Lone woman detected. See the attached image."
        else:
            return  # Do nothing if no relevant condition is met

        try:
            message = client.messages.create(
                body=message_body,
                media_url=[image_url],
                from_=from_whatsapp_number,
                to=to_whatsapp_number
            )
            print(f"WhatsApp message sent with SID: {message.sid}")
        except Exception as e:
            print(f"Error sending WhatsApp message: {e}")


    # Updated process_frame in violence_tracker.py
    def process_frame(self, frame, location):
        player_detections = self.detect_frame(frame)
        player_labels, player_genders, violence_against_women, male_count, female_count = self.classify_players(frame, player_detections)

        # Check for violence or specific scenarios
        if violence_against_women or female_count == 1:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            action_detected = "Violence against a woman" if violence_against_women else "Lone female detected"

            # Log detection results, including location
            self.log_violence_to_csv(timestamp, action_detected, male_count, female_count, location)

            # Save screenshot and send alerts
            self.save_screenshot(frame, violence_against_women, female_count,location)

        # Draw bounding boxes with gender and action labels
        frame, violence_detected, green_box_detected, male_count, female_count = self.draw_bboxes(
            frame, player_detections, [], player_labels, player_genders, False, male_count, female_count
        )

        return frame

