import cv2
import pyautogui
import json
from tkinter import Tk, Button, Label, Toplevel, Entry, OptionMenu, StringVar
import numpy as np

class RegionManager:
    def __init__(self):
        self.regions = []  # List to store regions
        self.region_locations = {}  # Dictionary to store region id and location
        self.json_file = "regions.json"

        # Create the main tkinter window
        self.root = Tk()
        self.root.title("Region Manager")

        # Add widgets
        Label(self.root, text="Add regions and locations").pack(pady=10)
        
        # Option for selecting region type
        Label(self.root, text="Select Region Configuration:").pack(pady=5)
        self.region_type = StringVar(self.root)
        self.region_type.set("1")  # Default is 1 region
        region_options = ["1", "2", "4", "6"]
        OptionMenu(self.root, self.region_type, *region_options).pack(pady=5)

        Button(self.root, text="Divide Screen", command=self.divide_screen_from_input).pack(pady=10)
        Button(self.root, text="Save and Quit", command=self.save_and_quit).pack(pady=5)

    def divide_screen_from_input(self):
        """Divide the screen based on selected region configuration."""
        region_type = self.region_type.get()
        screen_width, screen_height = pyautogui.size()

        if region_type == "1":
            # Full screen (1 region)
            self.regions = [(0, 0, screen_width, screen_height)]
        elif region_type == "2":
            # Vertically split screen into 2 regions
            self.regions = [(0, 0, screen_width // 2, screen_height), (screen_width // 2, 0, screen_width // 2, screen_height)]
        elif region_type == "4":
            # Divide screen into 4 regions (2x2 grid)
            region_width = screen_width // 2
            region_height = screen_height // 2
            self.regions = [
                (0, 0, region_width, region_height),
                (region_width, 0, region_width, region_height),
                (0, region_height, region_width, region_height),
                (region_width, region_height, region_width, region_height)
            ]
        elif region_type == "6":
            # Divide screen into 6 regions (3x3 grid)
            region_width = screen_width // 3
            region_height = screen_height // 3
            self.regions = [
                (0, 0, region_width, region_height),
                (region_width, 0, region_width, region_height),
                (region_width * 2, 0, region_width, region_height),
                (0, region_height, region_width, region_height),
                (region_width, region_height, region_width, region_height),
                (region_width * 2, region_height, region_width, region_height),
            ]

        # Display regions for confirmation
        for idx, region in enumerate(self.regions):
            print(f"Region {idx + 1}: {region}")

    def add_region(self):
        """Let the user select a region from the screen."""
        print("Select a region by dragging your mouse...")
        selected_region = self.select_region()
        if selected_region:
            region_id = len(self.regions) + 1
            self.regions.append(selected_region)
            print(f"Region {region_id} added: {selected_region}")
            self.get_location_from_user(region_id)
        else:
            print("No region selected.")

    def select_region(self):
        """Capture the screen and let the user drag to select a region."""
        frame = pyautogui.screenshot()
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        temp_frame = frame.copy()
        start_x, start_y, end_x, end_y = -1, -1, -1, -1
        region_selected = False

        def mouse_callback(event, x, y, flags, param):
            nonlocal start_x, start_y, end_x, end_y, region_selected, temp_frame
            if event == cv2.EVENT_LBUTTONDOWN:
                start_x, start_y = x, y
            elif event == cv2.EVENT_MOUSEMOVE and start_x != -1 and start_y != -1:
                temp_frame = frame.copy()
                cv2.rectangle(temp_frame, (start_x, start_y), (x, y), (0, 255, 0), 2)
                cv2.imshow("Select Region", temp_frame)
            elif event == cv2.EVENT_LBUTTONUP:
                end_x, end_y = x, y
                region_selected = True

        cv2.imshow("Select Region", frame)
        cv2.setMouseCallback("Select Region", mouse_callback)

        while not region_selected:
            cv2.imshow("Select Region", temp_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        if start_x != -1 and start_y != -1 and end_x != -1 and end_y != -1:
            return start_x, start_y, end_x - start_x, end_y - start_y
        return None

    def get_location_from_user(self, region_id):
        """Prompt the user to enter a location for the region."""
        top = Toplevel(self.root)
        top.title(f"Enter Location for Region {region_id}")

        Label(top, text="Enter location:").pack(padx=10, pady=10)
        location_entry = Entry(top)
        location_entry.pack(padx=10, pady=10)

        def on_submit():
            self.region_locations[region_id] = location_entry.get()
            top.destroy()

        Button(top, text="Submit", command=on_submit).pack(pady=5)

    def save_to_json(self):
        """Save regions and locations to a JSON file."""
        data = {
            "regions": self.regions,
            "locations": self.region_locations
        }
        with open(self.json_file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Regions and locations saved to {self.json_file}")

    def save_and_quit(self):
        """Save data and exit the application."""
        self.save_to_json()
        self.root.quit()

    def run(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()

if __name__ == "__main__":
    app = RegionManager()
    app.run()
