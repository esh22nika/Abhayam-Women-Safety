import streamlit as st
import subprocess
import os
import signal
import psutil
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from datetime import date
import plotly.express as px
import altair as alt
import streamlit as st
from streamlit_folium import st_folium
import folium
import json

# Constants
CREDENTIALS_FILE = "credentials.csv"

# Helper Functions
def verify_credentials(username, password):
    """Verify credentials."""
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, mode="r") as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == username and row[1] == password:
                    return True
    return False


def save_credentials(username, password):
    """Save credentials."""
    if not credentials_exist(username):
        with open(CREDENTIALS_FILE, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([username, password])
        return True
    return False


def credentials_exist(username):
    """Check if username exists."""
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, mode="r") as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == username:
                    return True
    return False


def start_processing():
    """Start main.py and 1.py for processing."""
    try:
        # Start main.py
        main_process = subprocess.Popen(["python", "main.py"])
        st.session_state["main_process"] = main_process

        # Start 1.py (Flask app)
        flask_process = subprocess.Popen(["python", "1.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        st.session_state["flask_process"] = flask_process

        # Notify user and provide a link to the Flask app
        st.success("Processing started successfully.")
        st.write("The Flask application is running.")
        st.markdown("[Open Flask App](http://127.0.0.1:5002)", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error starting processing: {e}")

def stop_processing():
    """Stop main.py, 1.py, and all associated processes."""
    try:
        # Stop main.py process
        if "main_process" in st.session_state and st.session_state["main_process"]:
            st.session_state["main_process"].terminate()
            st.session_state["main_process"] = None

        # Stop 1.py process (Flask app)
        if "flask_process" in st.session_state and st.session_state["flask_process"]:
            st.session_state["flask_process"].terminate()
            st.session_state["flask_process"] = None

        st.success("All processes stopped successfully.")
    except Exception as e:
        st.error(f"Error stopping processing: {e}")

def configuration():
    """Start region_manager.py for configuration."""
    try:
        # Start region_manager.py (Tkinter-based GUI)
        region_manager_process = subprocess.Popen(["python", "region_manager.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        st.session_state["region_manager_process"] = region_manager_process

        # Notify user
        st.success("Region Manager configuration started successfully.")
        st.write("The Region Manager GUI is running.")
    except Exception as e:
        st.error(f"Error starting Region Manager configuration: {e}")

def load_regions_json():
    try:
        with open("regions.json", "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        st.error("regions.json file not found.")
        return None

# Function to display regions and locations in a neat table
def display_regions_table():
    data = load_regions_json()
    
    if data:
        # Extract the regions and locations
        regions = data.get("regions", [])
        locations = data.get("locations", {})

        # Convert regions into a pandas DataFrame
        df_regions = pd.DataFrame(regions, columns=["X", "Y", "Width", "Height"])

        # Add a location column (if any locations are present)
        df_regions["Location"] = df_regions.index.map(lambda idx: locations.get(str(idx + 1), "N/A"))

        # Display the table using Streamlit's st.dataframe
        st.title("Regions Configuration Table")
        st.dataframe(df_regions)

# Function to add a back arrow to navigate to previous page
def back_arrow():
    if st.button("← Back", key="back"):
        # Reset the location and subfolder selections when the back button is clicked
        st.session_state.selected_location = None
        st.session_state.selected_subfolder = None
        st.session_state["one_female_ran"] = False # Optional: Reset the state of "one_female_ran"
        st.rerun()

def display_detected_images():
    st.title("Detected Images")
    
    # Back arrow for navigation
    back_arrow()

    # Run one_female.py and geminLabel.py only when "Detected Images" is selected and "one_female_ran" is not set
    if "one_female_ran" not in st.session_state or not st.session_state["one_female_ran"]:
        with st.spinner("Running detection scripts..."):
            try:
                # Run both scripts concurrently
                one_female_process = subprocess.Popen(["python", "one_female.py"])
                geminiLabel_process = subprocess.Popen(["python", "geminiLabel.py"])

                # Wait for both processes to finish
                one_female_process.wait()
                geminiLabel_process.wait()

                # Mark one_female.py as ran
                st.session_state["one_female_ran"] = True
                st.success("Detection completed successfully.")
            except Exception as e:
                st.error(f"Error running detection scripts: {e}")
                return

    # Display folders once the script has completed
    if "selected_location" not in st.session_state or st.session_state["selected_location"] is None:
        folders = [f for f in os.listdir() if os.path.isdir(f) and f != "__pycache__" and f != ".streamlit"]
        if folders:
            for folder in folders:
                if st.button(folder):
                    st.session_state["selected_location"] = folder
                    st.rerun()  # Rerun to show subfolders
        else:
            st.warning("No detected images found.")
    elif "selected_subfolder" not in st.session_state or st.session_state["selected_subfolder"] is None:
        location = st.session_state["selected_location"]
        subfolders = ["one_female", "violence_against_women", "gesture"]  # Customize as per your folder structure
        for subfolder in subfolders:
            if st.button(subfolder):
                st.session_state["selected_subfolder"] = subfolder
                st.rerun()  # Rerun to display images
    else:
        location = st.session_state["selected_location"]
        subfolder = st.session_state["selected_subfolder"]
        folder_path = os.path.join(location, subfolder)

        # Check if the subfolder exists before trying to display images
        if not os.path.exists(folder_path):
            st.warning(f"Subfolder '{subfolder}' does not exist under {location}.")
            return
        
        images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Display images or a message if no images are found
        if images:
            for img in images:
                img_path = os.path.join(folder_path, img)
                st.image(img_path, caption=img, use_container_width=True)
        else:
            st.warning(f"No images found in {subfolder} under {location}.")


# Hotspot Analytics Section
def hotspot_analytics():
    """Display and download hotspot analytics."""
    st.title("Hotspot Analytics")

    # Check if the CSV file exists
    file_path = "hotspot.csv"
    if os.path.exists(file_path):
        # Load CSV data
        df = pd.read_csv(file_path)

        # Display data in tabular form
        st.subheader("Hotspot Data")
        st.dataframe(df, use_container_width=True)

        # Download button for the CSV file
        st.download_button(
            label="Download Hotspot CSV",
            data=df.to_csv(index=False),
            file_name="hotspot.csv",
            mime="text/csv",
        )
    else:
        st.error("Hotspot CSV file not found.")



# Function to render the map with markers
def map_page():
    st.title("Interactive Map with Markers")

    # Define the latitude and longitude for the markers
    locations = [
        {"name": "Marker 1", "lat": 19.0219 ,"lon": 72.8450},  # Mumbai
        {"name": "Marker 2", "lat": 19.0197, "lon": 72.8466},  # Delhi
        {"name": "Marker 3", "lat": 19.0243, "lon": 72.8520},  # Bangalore
        {"name": "Marker 4", "lat": 19.0234, "lon": 72.8483},  # Kolkata
        {"name": "Marker 5", "lat": 19.0228, "lon": 72.8471},  # Chennai
    ]

    # Create a Folium map centered at an average location
    map_center = [20.5937, 78.9629]  # Center of India
    folium_map = folium.Map(location=map_center, zoom_start=5, tiles="OpenStreetMap")

    # Add markers to the map
    for loc in locations:
        folium.Marker(
            location=[loc["lat"], loc["lon"]],
            popup=f"<b>{loc['name']}</b>",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(folium_map)

    # Render the map using Streamlit
    st_data = st_folium(folium_map, width=700, height=500)

def threat_based_classification():
    # Your threat-based detection logic and UI go here.
    st.title("Threat Based Classification")
    
    # Display the kind of content you want (e.g., analysis, classification results, etc.)
    st.write("This page is for threat-based classification, here you can analyze threat levels based on certain parameters.")
    
    # For example: Show some results or models related to threat classification
    # You could also display figures, images, or graphs here.
    st.image("threat.png", caption="Threat Classification Example")

def general_analytics(violence_log_path, sos_gestures_path):
    # Load datasets
    violence_data = pd.read_csv(violence_log_path, parse_dates=["Timestamp"])
    sos_columns = ["Timestamp", "Location", "Image_URL"]
    sos_data = pd.read_csv(sos_gestures_path, header=None, names=sos_columns, parse_dates=["Timestamp"])

    # Sidebar filter: Select date
    min_date = date(2024, 12, 1)
    max_date = date(2024, 12, 7)

    # Adjust default date dynamically
    today = date.today()
    default_date = today if min_date <= today <= max_date else min_date

    selected_date = st.sidebar.date_input(
        "Select Date:",
        value=default_date,
        min_value=min_date,
        max_value=max_date
    )

    # Filter datasets based on selected date
    filtered_violence_data = violence_data[violence_data["Timestamp"].dt.date == selected_date]
    filtered_sos_data = sos_data[sos_data["Timestamp"].dt.date == selected_date]

    st.write(f"Showing data for: {selected_date}")

    # Visualization: Hourly Incidents on Selected Date
    hourly_violence_counts = filtered_violence_data.groupby(filtered_violence_data["Timestamp"].dt.hour).size()
    hourly_chart = alt.Chart(
        pd.DataFrame({"Hour": hourly_violence_counts.index, "Count": hourly_violence_counts.values})
    ).mark_bar(color="#EF553B", cornerRadius=10).encode(
        x=alt.X("Hour:O", title="Hour of Day"),
        y=alt.Y("Count:Q", title="Incident Count"),
        tooltip=["Hour", "Count"]
    ).properties(width=400, height=300, title="Hourly Violence Detection")

    # Monthly SOS Gesture Analysis
    sos_monthly_counts = sos_data.groupby(sos_data["Timestamp"].dt.strftime("%Y-%m")).size()
    fig_sos_monthly = px.bar(
        sos_monthly_counts,
        x=sos_monthly_counts.index,
        y=sos_monthly_counts.values,
        title="Monthly SOS Gesture Analysis",
        labels={"x": "Month", "y": "SOS Gesture Count"},
        color_discrete_sequence=["#636EFA"]
    )
    fig_sos_monthly.update_layout(
        template="plotly_white",
        margin=dict(t=40, b=20),
        title_font_size=18,
        height=300,
        width=400
    )

    # Violence Against One Woman
    violence_one_woman = filtered_violence_data[(filtered_violence_data["Male Count"] == 1) & (filtered_violence_data["Female Count"] == 1)]
    violence_one_woman_counts = violence_one_woman.groupby("Location").size()
    fig_violence_one_woman = px.bar(
        violence_one_woman_counts,
        x=violence_one_woman_counts.index,
        y=violence_one_woman_counts.values,
        title="Violence Against One Woman by Location",
        labels={"x": "Location", "y": "Incident Count"},
        color_discrete_sequence=["#AB63FA"]
    )
    fig_violence_one_woman.update_layout(
        template="plotly_white",
        margin=dict(t=40, b=20),
        title_font_size=18,
        height=300,
        width=400
    )

    # Gender Distribution
    total_male_count = violence_data["Male Count"].sum()
    total_female_count = violence_data["Female Count"].sum()
    gender_distribution = pd.DataFrame({
        "Gender": ["Male", "Female"],
        "Count": [total_male_count, total_female_count]
    })
    fig_gender_distribution = px.pie(
        gender_distribution,
        names="Gender",
        values="Count",
        title="Gender Distribution in Institution",
        hole=0.4,  # Creates a donut chart
        color_discrete_sequence=["#636EFA", "#EF553B"]
    )
    fig_gender_distribution.update_layout(
        template="plotly_white",
        margin=dict(t=40, b=20),
        title_font_size=18,
        height=300,
        width=400
    )

    # Modern Styling
    st.markdown(
        """
        <style>
        .card {
            background-color: rgba(255, 255, 255, 0.2);
            padding: 20px;
            margin: 10px;
            border-radius: 15px;
            box-shadow: 2px 2px 15px rgba(0,0,0,0.3);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Arrange visualizations with same size
    st.markdown("### Dashboard Overview")
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(fig_sos_monthly, use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(fig_violence_one_woman, use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.altair_chart(hourly_chart, use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(fig_gender_distribution, use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)


# Streamlit App
st.set_page_config(page_title="CCTV Monitoring", layout="wide")
st.markdown("""
    <style>
        .logo {
            position: fixed;
            top: 10px;
            right: 10px;
            z-index: 1000;
        }
    </style>
""", unsafe_allow_html=True)

# Make sure to use correct file path or URL for the logo
st.image("abhayamWhite.png", width=100)
st.title("Abhayam: Empowering Safety, Protecting Women.")

# Custom CSS for buttons and icons (Font Awesome icons used here)
st.markdown("""
    <style>
        .sidebar-button {
            display: block;
            padding: 10px;
            margin: 10px 0;
            background-color: #0073e6;
            color: white;
            border: none;
            border-radius: 5px;
            text-align: center;
            font-size: 16px;
            font-weight: bold;
        }
        .sidebar-button:hover {
            background-color: #005bb5;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar buttons with Font Awesome icons
st.sidebar.title("Navigation")

# Initialize `nav_option`
if "nav_option" not in st.session_state:
    st.session_state["nav_option"] = None

if st.sidebar.button("Login", key="login", help="Login to the system", use_container_width=True):
    st.session_state["nav_option"] = "Login"
elif st.sidebar.button("Signup", key="signup", help="Sign up for a new account", use_container_width=True):
    st.session_state["nav_option"] = "Signup"
elif st.sidebar.button("About Us", key="about_us", help="Learn more about us", use_container_width=True):
    st.session_state["nav_option"] = "About Us"
elif st.sidebar.button("Main Menu", key="main_menu", help="Go to main menu", use_container_width=True):
    st.session_state["nav_option"] = "Main Menu"
elif st.sidebar.button("Detected Images", key="detected_images", help="View detected images", use_container_width=True):
    st.session_state["nav_option"] = "Detected Images"
if st.sidebar.button("Hotspot Analytics", key="hotspot_analytics", help="View hotspot analytics", use_container_width=True):
    st.session_state["nav_option"] = "Hotspot Analytics"
if st.sidebar.button("General Analytics", key="general_analytics", help="View general analytics", use_container_width=True):
    st.session_state["nav_option"] = "General Analytics"
if st.sidebar.button("Hotspot Map", key="map_page", help="View Map analytics", use_container_width=True):
    st.session_state["nav_option"] = "Hotspot Map"
if st.sidebar.button("Threat Classification", key="threat_based_classification", help="View Threat Classification", use_container_width=True):
    st.session_state["nav_option"] = "Threat Classification"



# Initialize session state for login status if it doesn't exist
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

nav_option = st.session_state["nav_option"]

if nav_option == "Login":
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if verify_credentials(username, password):
            st.success("Login successful!")
            st.session_state["logged_in"] = True  # Set the login state
        else:
            st.error("Invalid username or password.")

elif nav_option == "Signup":
    st.subheader("Signup")
    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")
    if st.button("Signup"):
        if save_credentials(username, password):
            st.success("Signup successful!")
        else:
            st.error("Username already exists.")

elif nav_option == "About Us":
    st.title("About Us")
    st.subheader("Our Mission")
    st.write("Abhayam is a cutting-edge women safety analytics solution designed to enhance public safety by leveraging real-time surveillance and advanced analytical techniques. Our mission is to create safer environments for women and empower law enforcement agencies to act proactively in preventing crimes. Through Abhayam, we provide continuous monitoring and gender classification, ensuring that potential threats such as lone women at night, unusual gestures, or women surrounded by men are detected before escalation. By analyzing gender distribution, recognizing SOS gestures, and identifying hotspots, Abhayam offers valuable insights to help safeguard women in urban areas. Our system plays a crucial role in fostering a secure atmosphere and contributing to strategic safety planning, aiming to reduce crime and improve the safety of women everywhere.")
    st.subheader("Our Team")
    st.write("Our team consists of passionate developers dedicated to using AI for public safety.")
    st.subheader("Contact Us")
    st.write("You can reach us at eshanikaamballa@gmail.com")

elif nav_option == "Main Menu":
    if st.session_state["logged_in"]:
        st.subheader("Main Menu")

        # Start Processing Button
        if st.button("Start Processing"):
            start_processing()

        # Stop Processing Button
        if st.button("Stop Processing"):
            stop_processing()
        if st.button("Configuration"):
            configuration()
        if st.button("Show cam feed table"):
            load_regions_json()
            display_regions_table()

    else:
        st.error("Please log in to access this section.")

elif nav_option == "Detected Images":
    if st.session_state["logged_in"]:
        display_detected_images()
    else:
        st.error("Please log in to access this section.")

elif nav_option == "Hotspot Analytics":
    if st.session_state["logged_in"]:
        hotspot_analytics()
    else:
        st.error("Please log in to access this section.")

elif nav_option == "General Analytics":
    if st.session_state["logged_in"]:
        general_analytics('violence_log.csv','sos_gestures.csv')
    else:
        st.error("Please log in to access this section.")

elif nav_option == "Threat Classification":
    if st.session_state["logged_in"]:
        threat_based_classification()
    else:
        st.error("Please log in to access this section.")

elif st.session_state["logged_in"]:
    nav_option = st.sidebar.radio("Go to", [ "Hotspot Map"])

    if nav_option == "Hotspot Map":
        map_page()
else:
    st.sidebar.write("You need to log in first.")



