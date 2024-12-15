import json
import cv2
import pyautogui
import numpy as np
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import gesture
import violence_tracker
import queue

class RegionProcessor:
    def __init__(self, json_file="regions.json"):
        self.running = False
        self.json_file = json_file
        self.regions = []
        self.region_locations = {}

        # Load regions and locations from JSON
        self.load_from_json()

        # Load YOLO model once outside the loop
        self.yolo_model_path = "yolov8n.pt"
        self.tracker = violence_tracker.ViolenceTracker(yolo_model_path=self.yolo_model_path)

        # ThreadPoolExecutor to manage concurrent threads
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Queue for frame passing
        self.frame_queue = queue.Queue(maxsize=20)

    def load_from_json(self):
        """Load regions and locations from a JSON file."""
        try:
            with open(self.json_file, "r") as f:
                data = json.load(f)
                self.regions = data.get("regions", [])
                self.region_locations = data.get("locations", {})
            print(f"Loaded regions and locations from {self.json_file}")
        except FileNotFoundError:
            print(f"Error: {self.json_file} not found.")
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON.")

    def normalize_region(self, region):
        """Normalize the selected region."""
        x1, y1, width, height = region
        x_start = min(x1, x1 + width)
        y_start = min(y1, y1 + height)
        return x_start, y_start, abs(width), abs(height)

    def capture_screen(self, region):
        """Capture a screenshot of the specified region."""
        x, y, width, height = self.normalize_region(region)
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        return np.array(screenshot)

    def process_regions(self):
        """Start processing all selected regions."""
        self.running = True
        for i, region in enumerate(self.regions):
            region_id = i + 1
            location = self.region_locations.get(str(region_id), "Unknown")
            self.executor.submit(self.process_region_gesture, region, region_id, location)
            self.executor.submit(self.process_region_violence, region, region_id, location)

        self.display_frames()

    def process_region_gesture(self, region, region_id, location):
        """Process a single region for gesture detection."""
        gesture_start_time = None
        gesture_count = 0
        is_open = False
        SOS_THRESHOLD_COUNT = 3
        SOS_TIMEFRAME = 10

        while self.running:
            frame = self.capture_screen(region)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (640, 480))  # Resize for faster processing

            frame, gesture_start_time, gesture_count, is_open = gesture.process_frame_for_gesture(
                frame, gesture_start_time, gesture_count, is_open, SOS_THRESHOLD_COUNT, SOS_TIMEFRAME, location
            )

            try:
                self.frame_queue.put((f"Gesture Tracker - Region {region_id} - {location}", frame), timeout=0.1)
            except queue.Full:
                pass  # Ignore if the queue is full

    def process_region_violence(self, region, region_id, location):
        """Process a single region for violence detection."""
        while self.running:
            frame = self.capture_screen(region)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            processed_frame = self.tracker.process_frame(frame, location)

            processed_frame = cv2.resize(processed_frame, (640, 480))  # Resize for efficiency

            try:
                self.frame_queue.put((f"Violence Tracker - Region {region_id} - {location}", processed_frame), timeout=0.1)
            except queue.Full:
                pass  # Ignore if the queue is full

    def display_frames(self):
        """Display frames from the queue."""
        while self.running:
            try:
                window_name, frame = self.frame_queue.get(timeout=0.1)
                cv2.imshow(window_name, frame)
            except queue.Empty:
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

    def display_frames(self):
        """Display frames from the queue."""
        while self.running:
            try:
                window_name, frame = self.frame_queue.get(timeout=0.1)
                cv2.imshow(window_name, frame)
            except queue.Empty:
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

    def run(self):
        """Start processing regions."""
        if not self.regions:
            print("No regions to process. Please add regions using region_manager.py")
            return
        self.process_regions()

if __name__ == "__main__":
    processor = RegionProcessor()
    processor.run()
