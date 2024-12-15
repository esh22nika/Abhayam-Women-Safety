import time
import cv2
import mediapipe as mp
import cloudinary.uploader
import csv
from twilio.rest import Client
import os

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Configure Cloudinary
cloudinary.config(
    cloud_name="dbaacvp3e", 
    api_key="714437137622323", 
    api_secret="BLVWgl2D8b7b5Wjt_pMLoe5HdP4"
)

def send_whatsapp_alert(image_url):
    account_sid = "ACbe34242a5c03284f98c19d58bc38bb53"
    auth_token = "ecde757c677833f276cbf696af067bad"
    client = Client(account_sid, auth_token)

    from_whatsapp_number = "whatsapp:+14155238886"
    to_whatsapp_number = "whatsapp:+919372063867"

    try:
        message = client.messages.create(
            body="SOS Alert! Help required. See the attached image.",
            media_url=[image_url],
            from_=from_whatsapp_number,
            to=to_whatsapp_number
        )
        print(f"WhatsApp SOS alert sent with SID: {message.sid}")
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")

def trigger_sos_alert(frame, location):
    # Create location-based directories
    location_dir = f"{location}"
    gesture_subdir = os.path.join(location_dir, "gesture")
    os.makedirs(gesture_subdir, exist_ok=True)

    # Generate a unique filename using timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    local_path = os.path.join(gesture_subdir, f"sos_detected_{timestamp}.png")

    # Take a screenshot and save locally with the timestamp
    cv2.imwrite(local_path, frame)

    # Upload to Cloudinary
    try:
        upload_result = cloudinary.uploader.upload(local_path)
        image_url = upload_result.get("secure_url")
        print(f"Image uploaded to Cloudinary: {image_url}")

        # Send WhatsApp alert
        send_whatsapp_alert(image_url)

        # Save the SOS gesture and location in a CSV file
        with open("sos_gestures.csv", mode="a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), location, image_url])
        print(f"SOS gesture details saved to CSV.")

    except Exception as e:
        print(f"Error uploading to Cloudinary: {e}")

def shaka_gesture_detection(hand_landmarks):
    # Extract landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

    # MCP joints for folded finger detection
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]

    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # Conditions for the "Shaka" gesture
    thumb_pinky_distance = abs(thumb_tip.x - pinky_tip.x) + abs(thumb_tip.y - pinky_tip.y)
    folded_fingers = (
        index_tip.y > index_mcp.y and
        middle_tip.y > middle_mcp.y and
        ring_tip.y > ring_mcp.y
    )
    hand_upright = wrist.y > thumb_tip.y and wrist.y > pinky_tip.y

    # Check gesture
    if thumb_pinky_distance > 0.4 and folded_fingers and hand_upright:
        return True
    return False

def process_frame_for_gesture(frame, gesture_start_time, gesture_count, is_sos, SOS_THRESHOLD_COUNT, SOS_TIMEFRAME, location):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Shaka gesture detection logic
            is_current_sos = shaka_gesture_detection(hand_landmarks)

            # Initialize gesture start time if not set
            if gesture_start_time is None:
                gesture_start_time = time.time()

            # Track SOS gesture
            if is_current_sos and not is_sos:
                gesture_count += 1
                is_sos = True
            elif not is_current_sos and is_sos:
                is_sos = False

            # Handle SOS detection
            if time.time() - gesture_start_time > SOS_TIMEFRAME:
                if gesture_count >= SOS_THRESHOLD_COUNT:
                    # Trigger alert
                    trigger_sos_alert(frame, location)

                # Reset tracking
                gesture_start_time = None
                gesture_count = 0

    return frame, gesture_start_time, gesture_count, is_sos
