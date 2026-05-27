import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import font as tkfont, Button, filedialog
from PIL import Image, ImageTk
import threading
import cv2
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO
import tensorflow as tf
from scipy.spatial.distance import cdist
import time
import os
import requests
import json
from datetime import datetime
import re
from pymongo import MongoClient
from gridfs import GridFS
import pygame
import ttkbootstrap as tb

SEQ_LENGTH = 50
NUM_KEYPOINTS = 17
IMG_WIDTH, IMG_HEIGHT = 640, 480
MIN_KEYPOINTS = 5
MIN_CONFIDENCE = 0.3
DIST_THRESHOLD = 50 
MISSING_FRAMES_THRESHOLD = 5 
NECK = 1
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

class CameraCoverageDetector:
    def __init__(self, roi_scale=0.5):
        self.roi_scale = max(0.1, min(1.0, roi_scale))
        self.brightness_threshold = 20
        self.contrast_threshold = 15
        self.calibrated = False
        
    def _process_frame(self, frame):
        #Optimized frame processing with ROI
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        x1, y1 = int(w*(1-self.roi_scale)/2), int(h*(1-self.roi_scale)/2)
        x2, y2 = int(w*(1+self.roi_scale)/2), int(h*(1+self.roi_scale)/2)
        roi = gray[y1:y2, x1:x2]
        
        brightness = np.mean(roi)
        contrast = np.std(roi)
        return brightness, contrast
    
    def _auto_calibrate(self, brightness, contrast):
        #Set thresholds based on initial environment
        self.brightness_threshold = brightness * 0.3
        self.contrast_threshold = contrast * 0.4
        self.calibrated = True
            
    def check_coverage(self, frame):
        #Check if camera is covered
        brightness, contrast = self._process_frame(frame)
        
        if not self.calibrated and brightness > 5:
            self._auto_calibrate(brightness, contrast)
            return False
            
        if brightness < self.brightness_threshold and contrast < self.contrast_threshold:
            return True
        return False
            
class FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FeatureExtractor, self).__init__(**kwargs)
    
    def call(self, inputs):
        # Reshape to (batch, seq_len, 17, 3)
        keypoints = tf.reshape(inputs[:, :, :51], [-1, tf.shape(inputs)[1], 17, 3])
        
        # Velocity calculation
        velocity = keypoints[:, 1:, :, :2] - keypoints[:, :-1, :, :2]
        velocity = tf.pad(velocity, [[0,0], [1,0], [0,0], [0,0]])
        
        # Velocity features for key joints
        velocity_features = tf.stack([
            tf.norm(velocity[:, :, LEFT_WRIST], axis=-1),
            tf.norm(velocity[:, :, RIGHT_WRIST], axis=-1),
            tf.norm(velocity[:, :, LEFT_ANKLE], axis=-1),
            tf.norm(velocity[:, :, RIGHT_ANKLE], axis=-1),
            tf.norm(velocity[:, :, LEFT_KNEE], axis=-1),
            tf.norm(velocity[:, :, RIGHT_KNEE], axis=-1)
        ], axis=-1)
        
        # Relative motion
        torso_center = (keypoints[:, :, NECK, :2] + 
                       keypoints[:, :, LEFT_HIP, :2] + 
                       keypoints[:, :, RIGHT_HIP, :2]) / 3
        
        rel_motion = tf.stack([
            tf.norm(keypoints[:, :, LEFT_WRIST, :2] - torso_center, axis=-1),
            tf.norm(keypoints[:, :, RIGHT_WRIST, :2] - torso_center, axis=-1),
            tf.norm(keypoints[:, :, LEFT_ANKLE, :2] - torso_center, axis=-1),
            tf.norm(keypoints[:, :, RIGHT_ANKLE, :2] - torso_center, axis=-1)
        ], axis=-1)
        
        # Joint angles
        def calculate_angle(a, b, c):
            ba = a - b
            bc = c - b
            cosine_angle = tf.reduce_sum(ba * bc, axis=-1) / (
                tf.norm(ba, axis=-1) * tf.norm(bc, axis=-1) + 1e-6)
            return tf.acos(tf.clip_by_value(cosine_angle, -1.0, 1.0))
        
        angle_features = tf.stack([
            calculate_angle(keypoints[:, :, LEFT_SHOULDER, :2],
                          keypoints[:, :, LEFT_ELBOW, :2],
                          keypoints[:, :, LEFT_WRIST, :2]),
            calculate_angle(keypoints[:, :, RIGHT_SHOULDER, :2],
                          keypoints[:, :, RIGHT_ELBOW, :2],
                          keypoints[:, :, RIGHT_WRIST, :2]),
            calculate_angle(keypoints[:, :, LEFT_HIP, :2],
                          keypoints[:, :, LEFT_KNEE, :2],
                          keypoints[:, :, LEFT_ANKLE, :2]),
            calculate_angle(keypoints[:, :, RIGHT_HIP, :2],
                          keypoints[:, :, RIGHT_KNEE, :2],
                          keypoints[:, :, RIGHT_ANKLE, :2]),
            calculate_angle(keypoints[:, :, LEFT_SHOULDER, :2],
                          keypoints[:, :, NECK, :2],
                          keypoints[:, :, RIGHT_SHOULDER, :2])
        ], axis=-1)
        
        # Hand distance
        hand_dist = tf.norm(
            keypoints[:, :, LEFT_WRIST, :2] - keypoints[:, :, RIGHT_WRIST, :2],
            axis=-1, keepdims=True)
        
        # Combine all features (6 velocity + 4 relative_motion + 5 angles + 1 hand_distance = 16)
        features = tf.concat([velocity_features, rel_motion, angle_features, hand_dist], axis=-1)
        return tf.concat([inputs, features], axis=-1)
    
    def get_config(self):
        return super(FeatureExtractor, self).get_config()
    
class BehaviorDetectionWindow(tb.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Suspicious Behavior Detection")
        self.state("zoomed")
        self.resizable(True, True)
        self.configure(bg="#e8f5e9")  
        
        pygame.mixer.init()
        
        # Load alert sound
        self.alert_sound = None
        try:
            sound_path = os.path.join(os.path.dirname(__file__), "security_alarm.mp3")
            self.alert_sound = pygame.mixer.Sound(sound_path)
        except Exception as e:
            print(f"Error loading alert sound: {e}")
        
        # Alert settings
        self.last_alert_time = defaultdict(float)  
        self.alert_cooldown = 3.0    
        self.alert_playing = defaultdict(bool) 
        self.alert_channel = defaultdict(lambda: None) 
        
        self.is_camera_mode = False

        # MongoDB Connection
        try:
            self.mongo_client = MongoClient('mongodb://localhost:27017/')
            self.db = self.mongo_client['suspiciousBehaviorDatabase']
            self.fs = GridFS(self.db)
            
            # Create collections if they don't exist
            if 'behavior_incidents' not in self.db.list_collection_names():
                self.db.create_collection('behavior_incidents')
            if 'video_metadata' not in self.db.list_collection_names():
                self.db.create_collection('video_metadata')
            if 'analysis_results' not in self.db.list_collection_names():
                self.db.create_collection('analysis_results')
                
            print("Connected to MongoDB successfully")
        except Exception as e:
            print(f"Error connecting to MongoDB: {str(e)}")

        # Constants
        self.SEQ_LENGTH = 50
        self.NUM_KEYPOINTS = 17
        self.IMG_WIDTH, self.IMG_HEIGHT = 640, 480
        self.MIN_KEYPOINTS = 5
        self.MIN_CONFIDENCE = 0.3
        self.DIST_THRESHOLD = 50
        self.MISSING_FRAMES_THRESHOLD = 5
        self.yolo_det = None
        self.yolo_pose = None
        self.tcn_model = None
        self.tracker = None
        self.person_buffers = None  
        self.coverage_detector = None
        self.frame_count = 0
        self.fighting_incidents = defaultdict(list)  # {person_id: [(start_time, end_time)]}
        self.running_incidents = defaultdict(list)  # {person_id: [(start_time, end_time)]}
        self.loitering_incidents = defaultdict(list)  # {person_id: [(start_time, end_time)]}
        self.current_fighting = set()  # Track currently fighting IDs
        self.current_loitering = set()  # Track currently loitering IDs
        self.current_running = set()  # Track currently running IDs
        self.coverage_incidents = []  # List of [start_time, end_time] for camera coverage
        self.is_camera_covered = False  # Track current camera coverage state

        # Output folder for saving videos
        self.output_folder = "BehaviorOutputVideo"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Polygon drawing attributes
        self.current_points = []
        self.completed_polygons = []
        self.drawing_mode = False
        self.loitering_threshold = 30 
        self.person_times = {}  # Track entry times for each person
        self.people_in_polygons = set()  # Track people inside polygons

        self.paned_window = tb.Panedwindow(self, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        style = tb.Style()
        style.configure('light.TFrame', background='#e8f5e9')

        
        self.left_frame = tb.Frame(self.paned_window, style='light.TFrame')
        self.paned_window.add(self.left_frame, weight=1)

        self.right_frame = tb.Frame(self.paned_window, style='light.TFrame')
        self.paned_window.add(self.right_frame, weight=1)

        self.update()  
        window_width = self.winfo_width()
        self.paned_window.sashpos(0, window_width // 2)

        self.setup_left_frame()
        self.setup_polygon_controls()
        self.setup_right_frame()

        # Initialize video capture
        self.cap = None
        self.running = False
        self.detection_thread = None
        self.out = None
        self.video_source = None

        # Initialize models and trackers
        self.yolo_det = None
        self.yolo_pose = None
        self.tcn_model = None
        self.tracker = None
        self.person_buffers = None
        self.coverage_detector = None
        self.frame_count = 0
        self.tracker_config = {
            'tracker_type': 'bytetrack',
            'track_high_thresh': 0.5,
            'track_low_thresh': 0.1,
            'new_track_thresh': 0.6,
            'match_thresh': 0.8,
            'frame_rate': 30
        }

        # Load models in a separate thread
        threading.Thread(target=self.load_models, daemon=True).start()
        self.cleanup_output_files()

    def load_models(self):
        self.yolo_det = YOLO('yolov8n.pt')
        self.yolo_pose = YOLO('yolov8n-pose.pt')
        
        # Load the TCN model with custom FeatureExtractor layer
        custom_objects = {'FeatureExtractor': FeatureExtractor}
        self.tcn_model = tf.keras.models.load_model('trainedKJ4notAll2_model.keras', custom_objects=custom_objects)

        # Initialize tracker and buffers
        self.person_buffers = defaultdict(lambda: deque(maxlen=self.SEQ_LENGTH))
        self.coverage_detector = CameraCoverageDetector(roi_scale=0.6)
        
        # Update status
        self.update_results_text("Models loaded successfully")

    def setup_left_frame(self):
        label = tb.Label(self.left_frame, text="Suspicious Behavior Detection", font=("Helvetica", 16), bootstyle="primary", background="#e8f5e9", foreground="#004d40")
        label.pack(pady=10)

        self.canvas = tk.Canvas(self.left_frame, width=640, height=480, bg="black")
        self.canvas.pack(pady=10, padx=10)
        self.canvas.create_rectangle(0, 0, 640, 480, fill="black", outline="gray", width=2)

        self.button_frame = tb.Frame(self.left_frame, style='light.TFrame')
        self.button_frame.pack(pady=10)

        self.upload_button = tb.Button(
            self.button_frame,
            text="Upload Video",
            command=self.upload_video,
            bootstyle="primary",
            width=15
        )
        self.upload_button.pack(side=tk.LEFT, padx=10)

        self.camera_button = tb.Button(
            self.button_frame,
            text="Open Camera",
            command=self.open_camera,
            bootstyle="success",
            width=15
        )
        self.camera_button.pack(side=tk.LEFT, padx=10)

        self.close_camera_button = tb.Button(
            self.button_frame,
            text="Close Camera",
            command=self.close_camera,
            bootstyle="danger",
            width=15
        )
        self.close_camera_button.pack(side=tk.LEFT, padx=10)
        self.close_camera_button.pack_forget()

        self.end_video_button = tb.Button(
            self.button_frame,
            text="End Video",
            command=self.end_video,
            bootstyle="danger",
            width=15
        )
        self.end_video_button.pack(side=tk.LEFT, padx=10)
        self.end_video_button.pack_forget()

    def setup_polygon_controls(self):
        """Add polygon drawing controls to the left frame"""
        self.polygon_frame = tb.Frame(self.left_frame, style='light.TFrame')
        self.polygon_frame.pack(pady=5)
        
        self.draw_button = tb.Button(
            self.polygon_frame,
            text="Draw Polygon (D)",
            command=self.toggle_drawing_mode,
            bootstyle="info",
            width=15
        )
        self.draw_button.pack(side=tk.LEFT, padx=10)
        
        self.clear_button = tb.Button(
            self.polygon_frame,
            text="Clear Last (C)",
            command=self.clear_last_polygon,
            bootstyle="warning",
            width=15
        )
        self.clear_button.pack(side=tk.LEFT, padx=10)
        
        self.clear_all_button = tb.Button(
            self.polygon_frame,
            text="Clear All (X)",
            command=self.clear_all_polygons,
            bootstyle="danger",
            width=15
        )
        self.clear_all_button.pack(side=tk.LEFT, padx=10)
        
        # Hide polygon controls initially
        self.draw_button.pack_forget()
        self.clear_button.pack_forget()
        self.clear_all_button.pack_forget()
        
        # Bind keyboard shortcuts
        self.bind('<d>', lambda e: self.toggle_drawing_mode())
        self.bind('<c>', lambda e: self.clear_last_polygon())
        self.bind('<x>', lambda e: self.clear_all_polygons())
        
        # Bind mouse events for polygon drawing
        self.canvas.bind("<Button-1>", self.mouse_callback)
        self.canvas.bind("<Button-3>", self.mouse_callback)

    def setup_right_frame(self):
        self.right_frame.pack_propagate(True)
        results_label = tb.Label(
            self.right_frame,
            text="Detection Results",
            font=("Helvetica", 16),
            bootstyle="primary",
            background="#e8f5e9",
            foreground="#004d40"
        )
        results_label.pack(pady=10)

        style = tb.Style()
        style.configure('DarkGreen.Vertical.TScrollbar', troughcolor='#e8f5e9', bordercolor='#004d40', arrowcolor='#004d40', background='#004d40')

        text_frame = tb.Frame(self.right_frame, style='light.TFrame')
        text_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.results_text = tk.Text(
            text_frame,
            wrap=tk.WORD,
            width=50,
            height=20,
            font=("Helvetica", 12)
        )
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tb.Scrollbar(
            text_frame,
            command=self.results_text.yview,
            bootstyle="primary-round",
            style='DarkGreen.Vertical.TScrollbar'
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)

        self.api_button = tb.Button(
            self.right_frame,
            text="Analyze",
            command=self.send_to_api,
            bootstyle="info",
            width=15
        )

    def track_behavior(self, person_id, behavior_type, timestamp):
        if behavior_type == "fighting":
            if person_id not in self.current_fighting:
                # New fighting incident
                self.current_fighting.add(person_id)
                self.fighting_incidents[person_id].append([timestamp, None])
            else:
                # Update ongoing incident
                if self.fighting_incidents[person_id]:
                    self.fighting_incidents[person_id][-1][1] = timestamp
        elif behavior_type == "loitering":
            if person_id not in self.current_loitering:
                # New loitering incident
                self.current_loitering.add(person_id)
                self.loitering_incidents[person_id].append([timestamp, None])
            else:
                # Update ongoing incident
                if self.loitering_incidents[person_id]:
                    self.loitering_incidents[person_id][-1][1] = timestamp
        elif behavior_type == "running":
            if person_id not in self.current_running:
                # New running incident
                self.current_running.add(person_id)
                self.running_incidents[person_id].append([timestamp, None])
            else:
                # Update ongoing incident
                if self.running_incidents[person_id]:
                    self.running_incidents[person_id][-1][1] = timestamp

    def end_behavior_tracking(self, person_id, behavior_type, timestamp):
        if behavior_type == "fighting" and person_id in self.current_fighting:
            self.current_fighting.remove(person_id)
            if self.fighting_incidents[person_id]:
                self.fighting_incidents[person_id][-1][1] = timestamp
        elif behavior_type == "loitering" and person_id in self.current_loitering:
            self.current_loitering.remove(person_id)
            if self.loitering_incidents[person_id]:
                self.loitering_incidents[person_id][-1][1] = timestamp
        elif behavior_type == "running" and person_id in self.current_running:
            self.current_running.remove(person_id)
            if self.running_incidents[person_id]:
                self.running_incidents[person_id][-1][1] = timestamp

    def generate_summary(self):
        summary = "\n=== BEHAVIOR DETECTION SUMMARY ===\n"
        summary += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += "-"*40 + "\n"
        
        def format_time(seconds):
            if seconds is None:
                return "ongoing"
            mins, secs = divmod(int(seconds), 60)
            return f"{mins}:{secs:02d}"
        
        # Fighting incidents
        if self.fighting_incidents:
            summary += "FIGHTING INCIDENTS:\n"
            for pid, incidents in self.fighting_incidents.items():
                for i, (start, end) in enumerate(incidents, 1):
                    if end is not None and end >= start:
                        summary += f"  ID {pid} - Incident {i}: {format_time(start)} to {format_time(end)}\n"
        else:
            summary += "No fighting incidents detected\n"
        
        # Running incidents
        if self.running_incidents:
            summary += "\nRUNNING INCIDENTS:\n"
            for pid, incidents in self.running_incidents.items():
                for i, (start, end) in enumerate(incidents, 1):
                    if end is not None and end >= start:
                        summary += f"  ID {pid} - Incident {i}: {format_time(start)} to {format_time(end)}\n"
        else:
            summary += "\nNo running incidents detected\n"
            
        # Loitering incidents
        if self.loitering_incidents:
            summary += "\nLOITERING INCIDENTS:\n"
            for pid, incidents in self.loitering_incidents.items():
                for i, (start, end) in enumerate(incidents, 1):
                    duration = (end if end else time.time()) - start
                    summary += f"  ID {pid} - Incident {i}: {format_time(start)} to {format_time(end)} (Duration: {duration:.1f}s)\n"
        else:
            summary += "\nNo loitering incidents detected\n"
        
        # Camera coverage incidents
        if self.coverage_incidents:
            summary += "\nCAMERA COVERAGE INCIDENTS:\n"
            for i, (start, end, person_id) in enumerate(self.coverage_incidents, 1):
                if end is not None:
                    duration = end - start
                    id_info = f" (ID {person_id})" if person_id is not None else ""
                    summary += f"  Incident {i}{id_info}: {format_time(start)} to {format_time(end)} (Duration: {duration:.1f}s)\n"
                else:
                    id_info = f" (ID {person_id})" if person_id is not None else ""
                    summary += f"  Incident {i}{id_info}: {format_time(start)} to ongoing\n"
        else:
            summary += "\nNo camera coverage issues detected\n"
        
        summary += "="*40 + "\n"
        return summary

    def show_summary(self):
        self.last_summary = self.generate_summary()
        self.update_results_text(self.last_summary)
        
        # Store summary in MongoDB if not already stored
        if not hasattr(self, '_summary_stored'):
            self._summary_stored = True
        
    def remove_markdown(self, text):
        text = re.sub(r'#\s*', '', text)          
        text = re.sub(r'\*{1,2}', '', text)       
        text = re.sub(r'`+', '', text)             
        text = re.sub(r'>\s*', '', text)           
        text = re.sub(r'\n\s*-\s*', '\n', text)    
        return text

    def send_to_api(self):
        try:
            if not hasattr(self, 'last_summary'):
                self.update_results_text("\nNo summary available to send")
                return

            # Get the latest video_metadata_id and summary_id from MongoDB
            latest_video = self.db.video_metadata.find_one(sort=[('processing_date', -1)])
            latest_summary = self.db.behavior_incidents.find_one(sort=[('timestamp', -1)])
            
            video_metadata_id = latest_video['_id'] if latest_video else None
            summary_id = latest_summary['_id'] if latest_summary else None

            # Store all incident data before it gets cleared
            stored_incidents = {
                "fighting": [
                    {
                        "person_id": int(pid), 
                        "incidents": [
                            {"start": float(start), "end": float(end)}  
                            for start, end in incidents
                            if end is not None and end >= start
                        ]
                    }
                    for pid, incidents in self.fighting_incidents.items()
                ],
                "running": [
                    {
                        "person_id": int(pid),
                        "incidents": [
                            {"start": float(start), "end": float(end)}
                            for start, end in incidents
                            if end is not None and end >= start
                        ]
                    }
                    for pid, incidents in self.running_incidents.items()
                ],
                "loitering": [
                    {
                        "person_id": int(pid),
                        "incidents": [
                            {"start": float(start), "end": float(end)}
                            for start, end in incidents
                            if end is not None
                        ]
                    }
                    for pid, incidents in self.loitering_incidents.items()
                ],
                "camera_coverage": [
                    {
                        "start": float(start),
                        "end": float(end),
                        "person_id": int(pid) if pid is not None else None
                    }
                    for start, end, pid in self.coverage_incidents
                    if end is not None
                ]
            }

            # Create prompt for analysis
            prompt = f"""Please analyze this behavior detection summary. Here is the actual summary of detected incidents:

{self.last_summary}

And here is the structured data of these incidents:

{json.dumps(stored_incidents, indent=2)}

Please provide:
1. A detailed analysis of all detected behaviors and their timings
2. Analysis of any patterns or trends in the behavior (e.g., repeated incidents by the same ID)
3. Security recommendations based on the detected incidents
4. Risk assessment considering the frequency and severity of all incidents

Important: Focus on analyzing all incidents shown in the summary above."""

            # Format request for Groq API
            api_url = "https://api.groq.com/openai/v1/chat/completions"

            api_key = os.getenv("GROQ_API_KEY")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a security analysis expert. Your task is to analyze the behavior detection data. The text summary shows the actual detected incidents - analyze all types of incidents shown in the summary."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }

            response = requests.post(api_url, json=payload, headers=headers)
            
            if response.status_code == 200:
                # Display API response
                analysis = response.json()
                if analysis and "choices" in analysis and len(analysis["choices"]) > 0:
                    analysis_text = analysis["choices"][0]["message"]["content"]
                    clean_text = self.remove_markdown(analysis_text)
                    
                    self.clear_results_panel()
                    self.update_results_text("\nAnalysis Results:")
                    self.update_results_text(clean_text)
                    
                    try:
                        analysis_data = {
                            'timestamp': datetime.now(),
                            'summary_id': summary_id,
                            'video_metadata_id': video_metadata_id,
                            'analysis_text': clean_text,
                            'source': 'Groq API',
                            'model': 'llama-3.3-70b-versatile'
                        }
                        
                        result = self.db.analysis_results.insert_one(analysis_data)
                        self.update_results_text(f"\nAnalysis results stored in MongoDB with ID: {result.inserted_id}")
                    except Exception as e:
                        self.update_results_text(f"\nError storing analysis in MongoDB: {str(e)}")
                else:
                    self.update_results_text("\nNo analysis results in API response")
            else:
                self.update_results_text(f"\nError sending to API: {response.status_code}")
                self.update_results_text(response.text)

        except Exception as e:
            self.update_results_text(f"\nError preparing/sending data: {str(e)}")

    def toggle_drawing_mode(self):
        """Toggle polygon drawing mode and pause/resume video"""
        self.drawing_mode = not self.drawing_mode
        if self.drawing_mode:
            self.draw_button.config(text="Stop Drawing (D)", bg="#8e44ad")
            self.update_results_text("Drawing mode: ON - Left click to add points, Right click to complete")
            # Pause video when entering drawing mode
            self.paused = True
        else:
            self.draw_button.config(text="Draw Polygon (D)", bg="#9b59b6")
            self.update_results_text("Drawing mode: OFF")
            # Resume video when exiting drawing mode
            self.paused = False
            
    def clear_last_polygon(self):
        """Clear the last completed polygon"""
        if self.completed_polygons:
            self.completed_polygons.pop()
            self.update_results_text("Last polygon cleared")
        else:
            self.update_results_text("No polygons to clear")
            
    def clear_all_polygons(self):
        """Clear all polygons"""
        self.current_points = []
        self.completed_polygons = []
        self.person_times = {}
        self.update_results_text("All polygons cleared")
        
    def is_point_in_polygon(self, point, polygon_points):
        """Check if a point is inside a polygon"""
        return cv2.pointPolygonTest(
            contour=np.array(polygon_points),
            pt=point,
            measureDist=False
        ) >= 0
        
    def handle_polygon_drawing(self, frame):
        """Handle polygon drawing and loitering detection on frame"""
        frame_with_drawing = frame.copy()
        self.people_in_polygons = set()
        
        # Draw all completed polygons
        for polygon in self.completed_polygons:
            if len(polygon['points']) > 2:
                pts = np.array(polygon['points'], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame_with_drawing, [pts], True, (255, 0, 0), 2)  # Draw only the lines in blue
                
                # Display area information
                center = np.mean(pts, axis=0)[0].astype(int)
                cv2.putText(frame_with_drawing, f"Area: {polygon['area']:.0f}px", 
                          (center[0], center[1]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw current polygon in progress
        for i, point in enumerate(self.current_points):
            cv2.circle(frame_with_drawing, point, 5, (0, 0, 255), -1)  # Red points
            if i > 0:
                cv2.line(frame_with_drawing, self.current_points[i-1], point, (0, 255, 0), 2)  # Green lines
                
        # Check for loitering in polygons
        current_detections = set()
        if hasattr(self, 'person_buffers') and self.person_buffers:
            for person_id in self.person_buffers:
                if len(self.person_buffers[person_id]) > 0:
                    # Get latest keypoints for this person
                    latest_kpts = self.person_buffers[person_id][-1]
                    if latest_kpts.shape[0] == self.NUM_KEYPOINTS:
                        # Use hip keypoint (average of left and right hip)
                        left_hip = latest_kpts[LEFT_HIP]
                        right_hip = latest_kpts[RIGHT_HIP]
                        if left_hip[2] > self.MIN_CONFIDENCE and right_hip[2] > self.MIN_CONFIDENCE:
                            hip_center = (
                                int((left_hip[0] + right_hip[0])/2 * self.IMG_WIDTH),
                                int((left_hip[1] + right_hip[1])/2 * self.IMG_HEIGHT)
                            )
                            
                            current_detections.add(person_id)
                            
                            # Check if inside any polygon
                            inside = False
                            for poly in self.completed_polygons:
                                if self.is_point_in_polygon(hip_center, poly['points']):
                                    inside = True
                                    break
                                    
                            if inside:
                                self.people_in_polygons.add(person_id)
                            # Start or continue tracking time
                                if person_id not in self.person_times:
                                    self.person_times[person_id] = time.time()
                                    self.update_results_text(f"ID {person_id} entered monitored area")
                                
                                duration = time.time() - self.person_times[person_id]
                                
                                if duration >= self.loitering_threshold:
                                    # Draw loitering alert
                                    cv2.putText(frame_with_drawing, f"LOITERING {duration:.1f}s", 
                                            (hip_center[0], hip_center[1] - 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                            (0, 0, 255), 2)
                                    self.update_results_text(f"ALERT: ID {person_id} loitering for {duration:.1f}s")
                                    
                                    # Update the person's state to loitering
                                    for p_id, buffer in self.person_buffers.items():
                                        if p_id == person_id and buffer:
                                            # Get the latest keypoints
                                            latest_kpts = buffer[-1]
                                            if latest_kpts.shape[0] == self.NUM_KEYPOINTS:
                                                # Update the display text directly
                                                cv2.putText(frame_with_drawing, "LOITERING", 
                                                        (hip_center[0], hip_center[1] - 40),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                                        (0, 0, 255), 2)
                                                
        
        # Clean up person_times for people no longer detected
        self.person_times = {k: v for k, v in self.person_times.items() if k in current_detections}
        
        return frame_with_drawing
        
    def mouse_callback(self, event):
        """Handle mouse events for polygon drawing"""
        if not self.drawing_mode:
            return
            
        # Get mouse coordinates directly
        x, y = event.x, event.y
            
        if event.num == 1:  # Left click
            self.current_points.append((x, y))
            print(f"Added point at ({x}, {y})")
            self.update_results_text(f"Added point at ({x}, {y})")
            
        elif event.num == 3 and len(self.current_points) > 1:  # Right click to complete polygon
            if len(self.current_points) > 2:
                pts = np.array(self.current_points)
                area = cv2.contourArea(pts)
                
                self.completed_polygons.append({
                    'points': self.current_points.copy(),
                    'area': area
                })
                
                # Print polygon information
                print("\n=== New Polygon ===")
                print(f"Area: {area} px")
                print("Points (x,y):")
                for i, pt in enumerate(self.current_points):
                    print(f"  Point {i}: {pt}")
                print("===================")
                
                self.update_results_text(f"Completed polygon with area {area:.0f} px")
            self.current_points = []
            self.drawing_mode = False
            self.draw_button.config(text="Draw Polygon (D)", bg="#9b59b6")
            # Resume video after completing polygon
            self.paused = False

    def update_results_text(self, message):
        """Update the results text widget with a new message"""
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.results_text.update()

    def upload_video(self):
        # Reset camera mode flag
        self.is_camera_mode = False
        # Clear all tracking data before starting new video
        self.clear_all_tracking()
        self.clear_all_polygons()
        self.stop_camera()
        self.after(2000, self.clear_results_panel)
        self.update_results_text("Selecting video file...")
        
        file_path = filedialog.askopenfilename(
            parent=self,
            filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]
        )
        
        if file_path:
            self.video_source = file_path
            self.update_results_text(f"Processing video: {file_path}")
            self.start_video()
            self.end_video_button.pack(side=tk.LEFT, padx=10)
            self.draw_button.pack(side=tk.LEFT, padx=10)
            self.clear_button.pack(side=tk.LEFT, padx=10)
            self.clear_all_button.pack(side=tk.LEFT, padx=10)
            self.close_camera_button.pack_forget()
            self.api_button.pack_forget()
            self.upload_button.pack_forget()
            self.camera_button.pack_forget()

    def open_camera(self):
        self.is_camera_mode = True
        # Clear all tracking data before starting camera
        self.clear_all_tracking()
        self.clear_all_polygons()
        
        if hasattr(self, 'cap') and self.cap is not None:
            self.running = False
            self.cap.release()
            time.sleep(0.5)  
            self.cap = None
        if hasattr(self, 'out') and self.out is not None:
            self.out.release()
            time.sleep(0.5) 
            self.out = None
            
        self.after(2000, self.clear_results_panel)
        
        camera_indices = [0, 1, -1] 
        success = False
        
        for idx in camera_indices:
            try:
                self.update_results_text(f"Attempting to open camera index {idx}...")
                test_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) 
                
                if test_cap.isOpened():
                    ret, frame = test_cap.read()
                    if ret:
                        success = True
                        self.video_source = idx
                        test_cap.release()
                        time.sleep(0.5) 
                        break
                    else:
                        test_cap.release()
                        time.sleep(0.5)
                else:
                    if test_cap is not None:
                        test_cap.release()
                        time.sleep(0.5)
            except Exception as e:
                self.update_results_text(f"Error trying camera index {idx}: {str(e)}")
                if 'test_cap' in locals() and test_cap is not None:
                    test_cap.release()
                    time.sleep(0.5)
        
        if not success:
            self.update_results_text("Error: Could not access any camera. Please check your camera connection and permissions.")
            return
            
        try:
            self.start_video()
            self.close_camera_button.pack(side=tk.LEFT, padx=10)
            self.draw_button.pack(side=tk.LEFT, padx=10)
            self.clear_button.pack(side=tk.LEFT, padx=10)
            self.clear_all_button.pack(side=tk.LEFT, padx=10)
            self.end_video_button.pack_forget()
            self.api_button.pack_forget()
            self.upload_button.pack_forget()
            self.camera_button.pack_forget()
        except Exception as e:
            self.update_results_text(f"Error starting camera: {str(e)}")

    def close_camera(self):
        if hasattr(self, 'output_folder') and os.path.exists(self.output_folder):
            video_files = [f for f in os.listdir(self.output_folder) if f.endswith('.mp4')]
            if video_files:
                latest_video = os.path.join(self.output_folder, video_files[-1])
                video_metadata_id = self.store_video_in_mongodb(latest_video)
                
                # Generate and store summary
                self.show_summary()
                self.store_summary_in_mongodb(video_metadata_id)
                self._summary_stored = True 
        
        self.stop_camera()
        self.canvas.create_rectangle(0, 0, 640, 480, fill="black", outline="gray", width=2)
        
        self.end_video_button.pack_forget()
        self.draw_button.pack_forget()
        self.clear_button.pack_forget()
        self.clear_all_button.pack_forget()
        self.upload_button.pack(side=tk.LEFT, padx=10)
        self.camera_button.pack(side=tk.LEFT, padx=10)
        self.api_button.pack(pady=10)
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Camera closed")
        self.update_results_text("Camera closed - Data stored in MongoDB")
            
    def end_video(self):
        """End video processing and clean up"""
        # Store video in MongoDB
        if hasattr(self, 'output_folder') and os.path.exists(self.output_folder):
            video_files = [f for f in os.listdir(self.output_folder) if f.endswith('.mp4')]
            if video_files:
                latest_video = os.path.join(self.output_folder, video_files[-1])
                video_metadata_id = self.store_video_in_mongodb(latest_video)
                
                # Generate and store summary
                self.show_summary()
                self.store_summary_in_mongodb(video_metadata_id)
                self._summary_stored = True
        
        self.stop_camera()
        self.canvas.create_rectangle(0, 0, 640, 480, fill="black", outline="gray", width=2)
        
        self.end_video_button.pack_forget()
        self.draw_button.pack_forget()
        self.clear_button.pack_forget()
        self.clear_all_button.pack_forget()
        self.upload_button.pack(side=tk.LEFT, padx=10)
        self.camera_button.pack(side=tk.LEFT, padx=10)
        self.api_button.pack(pady=10)
        
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Video processing ended")
        self.update_results_text("Video processing ended - Data stored in MongoDB")

    def start_video(self):
        if self.cap:
            self.running = False
            self.cap.release()

        # Clear all tracking data before starting new video
        self.clear_all_tracking()
        self.cleanup_output_files()
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            self.update_results_text("Error: Cannot open video source")
            return

        self.running = True
        self.detection_thread = threading.Thread(target=self.process_video, daemon=True)
        self.detection_thread.start()
        self.update_video()
    
    def cleanup_output_files(self):
        """Clean up old temporary files but keep processed videos"""
        if os.path.exists(self.output_folder):
            for f in os.listdir(self.output_folder):
                if f.endswith('.tmp'): 
                    try:
                        os.remove(os.path.join(self.output_folder, f))
                    except Exception as e:
                        print(f"Error deleting temporary file {f}: {e}")

    def clean_up_tracking(self):
        """Clean up tracking data for people who left polygons"""
        current_frame_ids = {p['id'] for p in self.current_people} if hasattr(self, 'current_people') else set()
        
        # Remove people who are no longer detected
        for pid in list(self.person_times.keys()):
            if pid not in current_frame_ids:
                self.person_times.pop(pid, None)
        
        # Clean up people_in_polygons set
        self.people_in_polygons = {pid for pid in self.people_in_polygons if pid in current_frame_ids}
    
    def clean_up_buffers(self, current_track_ids):
        # Remove buffers for people no longer being tracked
        all_buffer_ids = set(self.person_buffers.keys())
        active_ids = set(current_track_ids)
        
        # Find IDs that are no longer present
        removed_ids = all_buffer_ids - active_ids
        
        # Remove their buffers
        for buffer_id in removed_ids:
            if buffer_id in self.person_buffers:  # Check if ID exists before deleting
                print(f"Removing buffer for ID {buffer_id} (person no longer in frame)")
                self.person_buffers.pop(buffer_id, None)  # Using pop with default None to safely remove
        
        print(f"Current tracked IDs: {current_track_ids}")
        print(f"Buffer IDs: {list(self.person_buffers.keys())}")
    
    def clear_results_panel(self):
        """Clear all text from the results panel"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.update()
    
    def clear_all_tracking(self):
        """Clear all tracking IDs and buffers"""
        if hasattr(self, 'person_buffers'):
            all_ids = list(self.person_buffers.keys())
            
            if all_ids:
                print(f"Clearing all tracking data. Removing IDs: {all_ids}")
                self.update_results_text(f"Clearing all tracking data. Removing IDs: {all_ids}")
                
            self.person_buffers.clear()
            self.person_times.clear()
            
            # Clear behavior tracking data
            self.fighting_incidents.clear()
            self.running_incidents.clear()
            self.loitering_incidents.clear()
            self.current_fighting.clear()
            self.current_running.clear()
            self.current_loitering.clear()
            self.coverage_incidents.clear()
            self.is_camera_covered = False
            
            print("All tracking data cleared")
            self.update_results_text("All tracking data cleared")

    def process_video(self):
        self.frame_count = 0
        self.paused = False  # Initialize paused state
        last_frame = None  # Store last frame for paused state
        
        # Set up video writer
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0: 
            fps = 30.0  
            
        if isinstance(self.video_source, str):
            video_name = os.path.basename(self.video_source)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = f"camera_recording_{timestamp}.mp4"
            
        output_video_path = os.path.join(self.output_folder, video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_video_path, fourcc, fps, (self.IMG_WIDTH, self.IMG_HEIGHT))

        self.current_fps = fps

        COLOR_NORMAL = (0, 255, 0)  # Green
        COLOR_RUNNING = (0, 165, 255)  # Orange
        COLOR_FIGHTING = (0, 0, 255)  # Red

        while self.running:
            if not self.paused:
                start_time = time.time()
                ret, frame = self.cap.read()
                self.frame_count += 1
                
                if not ret:
                    if self.video_source != 0:  # Not a camera
                        self.update_results_text("Video processing completed")
                    break
                
                # Store the current frame
                last_frame = frame.copy()
            else:
                # Use the last frame when paused
                if last_frame is None:
                    continue
                frame = last_frame.copy()
            
            # Resize frame to our processing dimensions
            frame = cv2.resize(frame, (self.IMG_WIDTH, self.IMG_HEIGHT))
            
            # Calculate and display FPS only when not paused
            if not self.paused:
                time_diff = time.time() - start_time
                if time_diff > 0:  # Prevent division by zero
                    fps = 1/time_diff
                else:
                    fps = 0
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            # Check for camera coverage only when not paused
            if not self.paused:
                camera_covered = self.coverage_detector.check_coverage(frame)
                if camera_covered:
                    if not self.is_camera_covered:  # Camera just got covered
                        self.is_camera_covered = True
                        self.coverage_incidents.append([self.frame_count/self.current_fps, None, None])
                        if self.is_camera_mode:
                            self.play_alert_sound("coverage")
                    self.update_results_text("Warning: Camera may be covered")
                    frame = self.draw_camera_covered(frame)
                elif self.is_camera_covered:  # Camera just got uncovered
                    self.is_camera_covered = False
                    if self.coverage_incidents and self.coverage_incidents[-1][1] is None:
                        self.coverage_incidents[-1][1] = self.frame_count/self.current_fps
                    self.stop_alert_sound("coverage")
                
                if camera_covered:
                    self.update_gui(frame)
                    # Write the frame to output video
                    self.out.write(frame)
                    continue
            
            # Only process detections when not paused
            if not self.paused:
                # First detect people with YOLOv8n
                det_results = self.yolo_det(frame, verbose=False)
                boxes = det_results[0].boxes.xyxy.cpu().numpy() if det_results[0].boxes else []
                classes = det_results[0].boxes.cls.cpu().numpy() if det_results[0].boxes else []
                confs = det_results[0].boxes.conf.cpu().numpy() if det_results[0].boxes else []
                
                # Filter for person class (class 0 in COCO) and confidence > 0.5
                person_boxes = []
                person_ids=[]
                for box, cls, conf in zip(boxes, classes, confs):
                    if cls == 0 and conf > 0.7:
                        person_boxes.append(box)
                
                # Get pose keypoints for behavior analysis only
                people_data, frame = self.extract_keypoints(frame)
                
                # Initialize valid_people list
                valid_people = []
                
                # Process behavior detection only when not paused
                if people_data:
                    # Filter out people with incomplete keypoints
                    for person in people_data:
                        keypoints = person['keypoints']
                        
                        if np.isnan(keypoints).any() or keypoints.shape[0] != self.NUM_KEYPOINTS:
                            continue
                        
                        valid_keypoints = np.sum(keypoints[:, 2] > self.MIN_CONFIDENCE)
                        if valid_keypoints < self.MIN_KEYPOINTS:
                            continue
                        
                        valid_people.append(person)
                    
                    person_ids = [p['id'] for p in valid_people]
                    # Update tracks (will handle people leaving frame)
                    print(f"Current tracked IDs: {[p['id'] for p in valid_people]}")
                    print(f"Buffer IDs: {list(self.person_buffers.keys())}")
                    self.update_results_text(f"Frame {self.frame_count}: Tracking {len(valid_people)} people")
                    self.clean_up_buffers(person_ids)
                    
                    # Process each person
                    for person in valid_people:
                        person_id = person['id']
                        keypoints_color = (230, 216, 173)
                        
                        # Draw keypoints
                        for kp in person['keypoints']:
                            x, y, conf = int(kp[0] * self.IMG_WIDTH), int(kp[1] * self.IMG_HEIGHT), kp[2]
                            if conf > self.MIN_CONFIDENCE:
                                cv2.circle(frame, (x, y), 3, keypoints_color, -1)
                        
                        # Store keypoints for tracking
                        if (person['keypoints'].shape[0] == self.NUM_KEYPOINTS and 
                            not np.isnan(person['keypoints']).any()):
                            self.person_buffers[person_id].append(person['keypoints'])
                        
                        # Check if person is inside any polygon using bottom center and center points
                        is_inside_polygon = False
                        if self.completed_polygons:
                            # Calculate person's bounding box
                            kpts = person['keypoints']
                            valid_kpts = kpts[kpts[:, 2] > self.MIN_CONFIDENCE]
                            if len(valid_kpts) > 0:
                                x_coords = valid_kpts[:, 0] * self.IMG_WIDTH
                                y_coords = valid_kpts[:, 1] * self.IMG_HEIGHT
                                x1, y1 = int(min(x_coords)), int(min(y_coords))
                                x2, y2 = int(max(x_coords)), int(max(y_coords))
                                
                                # Calculate bottom center and center points
                                bottom_center = (int((x1 + x2)/2), y2)
                                box_center = (int((x1 + x2)/2), int((y1 + y2)/2))
                                
                                for poly in self.completed_polygons:
                                    if (self.is_point_in_polygon(bottom_center, poly['points']) or 
                                        self.is_point_in_polygon(box_center, poly['points'])):
                                        is_inside_polygon = True
                                        break
                        
                        if person_id in self.people_in_polygons:
                            # Person is in polygon - only track time, no behavior detection
                            person['behavior'] = "tracking"
                            person['box_color'] = (255, 255, 0)  
                            
                            # Track time but don't update results panel unless loitering
                            if person_id not in self.person_times:
                                self.person_times[person_id] = time.time()
                                self.update_results_text(f"ID {person_id} entered monitored area")
                            
                            duration = time.time() - self.person_times[person_id]
                            person['display_text'] = f"TRACKING {duration:.1f}s"
                            
                            if duration >= self.loitering_threshold:
                                person['display_text'] = f"LOITERING {duration:.1f}s"
                                person['box_color'] = (0, 0, 255)  # Red for loitering
                                self.update_results_text(f"ALERT: ID {person_id} loitering for {duration:.1f}s")
                                if self.is_camera_mode:
                                    self.play_alert_sound("loitering")
            
                        elif len(self.person_buffers[person_id]) >= self.SEQ_LENGTH:
                            model_input = self.prepare_sequence(self.person_buffers[person_id])
                            if model_input is not None:
                                try:
                                    prediction = self.tcn_model.predict(model_input, verbose=0)
                                    behavior_scores = {
                                        "walking": prediction[0][0],
                                        "running": prediction[0][1],
                                        "fighting": prediction[0][2],
                                        "standing": prediction[0][3]
                                    }
                                    
                                    # Get the behavior with highest score
                                    max_behavior = max(behavior_scores, key=behavior_scores.get)
                                    max_score = behavior_scores[max_behavior]
                                    
                                    has_full_body = all(keypoints[kp, 2] > self.MIN_CONFIDENCE for kp in [NECK, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE])
                        
                                    # Determine final behavior and color
                                    final_behavior = "normal"
                                    box_color = COLOR_NORMAL
                                    display_text = "NORMAL"
                                    
                                    # Only update behavior and send alerts if person is outside all polygons
                                    if not any(self.is_point_in_polygon(bottom_center, poly['points']) or 
                                             self.is_point_in_polygon(box_center, poly['points']) 
                                             for poly in (self.completed_polygons or [])):
                                        if max_behavior == "fighting" and max_score > 0.9 and has_full_body and person_id not in self.people_in_polygons:
                                            final_behavior = "fighting"
                                            box_color = COLOR_FIGHTING
                                            display_text = "FIGHTING"
                                            self.track_behavior(person_id, "fighting", self.frame_count/self.current_fps)
                                            self.update_results_text(f"ALERT: Fighting detected (ID {person_id})")
                                            if self.is_camera_mode:
                                                self.play_alert_sound("fighting")
                                        elif max_behavior == "running" and max_score > 0.9 and has_full_body and person_id not in self.people_in_polygons:
                                            final_behavior = "running"
                                            box_color = COLOR_RUNNING
                                            display_text = "RUNNING"
                                            self.track_behavior(person_id, "running", self.frame_count/self.current_fps)
                                            self.update_results_text(f"Alert: Running detected (ID {person_id})")
                                            if self.is_camera_mode:
                                                self.play_alert_sound("running")
                                        else:
                                            # End behavior tracking and stop alerts if they were previously active
                                            if person_id in self.current_fighting:
                                                self.end_behavior_tracking(person_id, "fighting", self.frame_count/self.current_fps)
                                                if len(self.current_fighting) == 0:  # Only stop if no one else is fighting
                                                    self.stop_alert_sound("fighting")
                                            if person_id in self.current_running:
                                                self.end_behavior_tracking(person_id, "running", self.frame_count/self.current_fps)
                                                if len(self.current_running) == 0:  # Only stop if no one else is running
                                                    self.stop_alert_sound("running")
                                    
                                    # Store results
                                    person['behavior'] = final_behavior
                                    person['behavior_text'] = f'ID {person_id}: {display_text}'
                                    person['box_color'] = box_color
                                    person['display_text'] = display_text
                                except Exception as e:
                                    self.update_results_text(f"Prediction error for ID {person_id}: {e}")
                        else:
                            # Default to normal behavior when no polygons exist
                            person['behavior'] = "normal"
                            person['behavior_text'] = f'ID {person_id}: NORMAL'
                            person['box_color'] = COLOR_NORMAL
                            person['display_text'] = "NORMAL"

                # Second pass: Draw all boxes with behavior text
                for box in person_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    # Calculate bottom center and center points
                    bottom_center = (int((x1 + x2)/2), y2)  # Bottom center of bounding box
                    box_center = (int((x1 + x2)/2), int((y1 + y2)/2))  # Center of bounding box
                    
                    # Find the closest person for ID and behavior
                    min_dist = float('inf')
                    closest_person = None
                    for person in valid_people:
                        person_centroid = person['centroid']
                        dist = np.linalg.norm(np.array(box_center) - person_centroid)
                        if dist < min_dist:
                            min_dist = dist
                            closest_person = person
                    
                    # Default to NORMAL if no valid person found or keypoints not valid
                    color = (0, 255, 0)  # Default to green
                    text = "NORMAL"
                    
                    # Only show behavior if we found a close person with valid keypoints
                    if closest_person is not None and min_dist < 100:
                        person_id = closest_person['id']
                        
                        # Check if person is inside any polygon using bottom center point
                        is_inside = False
                        if self.completed_polygons:
                            for poly in self.completed_polygons:
                                # Check both bottom center and center points
                                if (self.is_point_in_polygon(bottom_center, poly['points']) or 
                                    self.is_point_in_polygon(box_center, poly['points'])):
                                    is_inside = True
                                    break
                        
                        if is_inside:
                            # Person is inside a polygon - mark as tracking
                            color = (255, 255, 0)  # Yellow for tracking
                            
                            # Start or continue tracking time
                            if person_id not in self.person_times:
                                self.person_times[person_id] = time.time()
                                self.update_results_text(f"ID {person_id} entered monitored area")
                            
                            duration = time.time() - self.person_times[person_id]
                            if duration >= self.loitering_threshold:
                                text = f"LOITERING {duration:.1f}s"
                                color = (0, 0, 255)  # Red for loitering
                                self.track_behavior(person_id, "loitering", duration)
                                self.update_results_text(f"ALERT: ID {person_id} loitering for {duration:.1f}s")
                                if self.is_camera_mode:
                                    self.play_alert_sound("loitering")
                            else:
                                text = f"TRACKING {duration:.1f}s"  # Show tracking duration
                                self.end_behavior_tracking(person_id, "loitering", duration)
                                self.stop_alert_sound("loitering")
                        
                        elif not self.completed_polygons:
                            # Only show behavior if no polygons exist
                            if 'behavior' in closest_person:
                                if closest_person['behavior'] == "fighting":
                                    color = (0, 0, 255)  # Red
                                    text = "FIGHTING"
                                elif closest_person['behavior'] == "running":
                                    color = (0, 165, 255)  # Orange
                                    text = "RUNNING"
                                elif closest_person['behavior'] == "normal":
                                    color = (0, 255, 0)  # Green
                                    text = "NORMAL"
                        
                        # Draw ID text
                        cv2.putText(frame, f"ID:{person_id}", (x1, y1 - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        # Draw center points for debugging
                        cv2.circle(frame, bottom_center, 3, (0, 0, 255), -1) 
                        cv2.circle(frame, box_center, 3, (255, 0, 0), -1) 
                    
                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw behavior text
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                    cv2.putText(
                        frame, 
                        text,
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (255, 255, 255),  # White text
                        2
                    )
            
            if self.drawing_mode or self.completed_polygons:
                frame = self.handle_polygon_drawing(frame)
            
            if self.paused:
                cv2.putText(frame, "PAUSED - Drawing Mode", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            self.out.write(frame)
            
            self.update_gui(frame)

        self.cap.release()
        if self.out:
            self.out.release()
            self.out = None
        if self.video_source == 0:
            self.update_results_text("Camera feed stopped")
        else:
            self.update_results_text("Video processing completed")

    def draw_camera_covered(self, frame):
        cv2.putText(frame, "CAMERA COVERED!", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    def extract_keypoints(self, frame):
        results = self.yolo_pose.track(
            frame, 
            persist=True, 
            verbose=False,
            tracker="bytetrack.yaml"  
        )
        
        people_data = []
        
        if results[0].boxes is None or results[0].boxes.id is None:
            return [], frame
        
        # Get tracked IDs
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        
        # Get keypoints and confidences
        kpts = results[0].keypoints.xy.cpu().numpy()
        confs = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints.conf is not None else np.ones((kpts.shape[0], self.NUM_KEYPOINTS))
        
        for i, (person_kpts, person_confs, track_id) in enumerate(zip(kpts, confs, track_ids)):
            if person_kpts.shape[0] != self.NUM_KEYPOINTS:
                continue
                
            person_data = np.concatenate([person_kpts, person_confs.reshape(-1, 1)], axis=-1)
            centroid = self.get_centroid(person_data)
            
            if centroid is not None and np.count_nonzero(person_data[..., 2] > self.MIN_CONFIDENCE) >= self.MIN_KEYPOINTS:
                person_data = self.normalize_keypoints(person_data, self.IMG_WIDTH, self.IMG_HEIGHT)
                people_data.append({
                    'id': track_id,
                    'keypoints': person_data,
                    'centroid': centroid,
                    'color': tuple(np.random.randint(0, 255, 3).tolist()),
                    'behavior_state': {'walking': 0.5, 'running': 0.5, 'fighting': 0.5, 'standing': 0.5}
                })
        
        return people_data, frame

    def smooth_keypoints(self, keypoints, prev_keypoints, alpha=0.4):
        if prev_keypoints is None:
            return keypoints
        return alpha * keypoints + (1 - alpha) * prev_keypoints

    def get_centroid(self, keypoints):
        visible = keypoints[keypoints[..., 2] > self.MIN_CONFIDENCE]
        if len(visible) >= self.MIN_KEYPOINTS:
            return np.mean(visible[..., :2], axis=0)
        return None

    def normalize_keypoints(self, keypoints,img_width, img_height):
        keypoints = keypoints.copy().astype(np.float32)
        keypoints[..., 0] /= float(self.IMG_WIDTH)
        keypoints[..., 1] /= float(self.IMG_HEIGHT)
        return np.nan_to_num(keypoints)

    def prepare_sequence(self, keypoints_buffer):
        if len(keypoints_buffer) < self.SEQ_LENGTH:
            return None
        
        # Convert to numpy array (50,17,3)
        try:
            sequence = np.array(list(keypoints_buffer)[-self.SEQ_LENGTH:])
            if sequence.shape[1] != self.NUM_KEYPOINTS:
                return None
        except Exception as e:
            self.update_results_text(f"Error preparing sequence: {e}")
            return None
        
        # Check if we have enough valid keypoints
        valid_frames = 0
        for frame in sequence:
            visible_kpts = np.sum(frame[:, 2] > self.MIN_CONFIDENCE)
            if visible_kpts >= self.MIN_KEYPOINTS:
                valid_frames += 1
        
        if valid_frames < self.SEQ_LENGTH * 0.8:
            return None
        
        # Compute all features (50,67)
        features = self.compute_features(sequence)
        
        # Add batch dimension (1,50,67)
        return features[np.newaxis, ...]

    def compute_features(self, sequence):
        # Reshape to (seq_len, 17, 3)
        keypoints = sequence.reshape(sequence.shape[0], self.NUM_KEYPOINTS, 3)
        
        # 1. Raw keypoints (flattened)
        raw_features = sequence.reshape(sequence.shape[0], -1)  # (seq_len, 51)
        
        # 2. Velocity features
        velocity = np.zeros_like(keypoints[..., :2])
        velocity[1:] = keypoints[1:, :, :2] - keypoints[:-1, :, :2]
        velocity_features = np.stack([
            np.linalg.norm(velocity[:, LEFT_WRIST], axis=-1),
            np.linalg.norm(velocity[:, RIGHT_WRIST], axis=-1),
            np.linalg.norm(velocity[:, LEFT_ANKLE], axis=-1),
            np.linalg.norm(velocity[:, RIGHT_ANKLE], axis=-1),
            np.linalg.norm(velocity[:, LEFT_KNEE], axis=-1),
            np.linalg.norm(velocity[:, RIGHT_KNEE], axis=-1)
        ], axis=-1)  # (seq_len, 6)
        
        # 3. Relative motion features
        torso_center = (keypoints[:, NECK, :2] + 
                       keypoints[:, LEFT_HIP, :2] + 
                       keypoints[:, RIGHT_HIP, :2]) / 3
        rel_motion = np.stack([
            np.linalg.norm(keypoints[:, LEFT_WRIST, :2] - torso_center, axis=-1),
            np.linalg.norm(keypoints[:, RIGHT_WRIST, :2] - torso_center, axis=-1),
            np.linalg.norm(keypoints[:, LEFT_ANKLE, :2] - torso_center, axis=-1),
            np.linalg.norm(keypoints[:, RIGHT_ANKLE, :2] - torso_center, axis=-1)
        ], axis=-1)  # (seq_len, 4)
        
        # 4. Joint angles
        def calculate_angle(a, b, c):
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        angle_features = np.zeros((sequence.shape[0], 5))
        for i in range(sequence.shape[0]):
            # Left arm angle
            if all(k < self.NUM_KEYPOINTS for k in [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]):
                a = keypoints[i, LEFT_SHOULDER, :2]
                b = keypoints[i, LEFT_ELBOW, :2]
                c = keypoints[i, LEFT_WRIST, :2]
                angle_features[i, 0] = calculate_angle(a, b, c)
            
            # Right arm angle
            if all(k < self.NUM_KEYPOINTS for k in [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]):
                a = keypoints[i, RIGHT_SHOULDER, :2]
                b = keypoints[i, RIGHT_ELBOW, :2]
                c = keypoints[i, RIGHT_WRIST, :2]
                angle_features[i, 1] = calculate_angle(a, b, c)
            
            # Left leg angle
            if all(k < self.NUM_KEYPOINTS for k in [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]):
                a = keypoints[i, LEFT_HIP, :2]
                b = keypoints[i, LEFT_KNEE, :2]
                c = keypoints[i, LEFT_ANKLE, :2]
                angle_features[i, 2] = calculate_angle(a, b, c)
            
            # Right leg angle
            if all(k < self.NUM_KEYPOINTS for k in [RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]):
                a = keypoints[i, RIGHT_HIP, :2]
                b = keypoints[i, RIGHT_KNEE, :2]
                c = keypoints[i, RIGHT_ANKLE, :2]
                angle_features[i, 3] = calculate_angle(a, b, c)
            
            # Shoulder angle
            if all(k < self.NUM_KEYPOINTS for k in [LEFT_SHOULDER, NECK, RIGHT_SHOULDER]):
                a = keypoints[i, LEFT_SHOULDER, :2]
                b = keypoints[i, NECK, :2]
                c = keypoints[i, RIGHT_SHOULDER, :2]
                angle_features[i, 4] = calculate_angle(a, b, c)
        
        # 5. Hand distance
        hand_dist = np.linalg.norm(
            keypoints[:, LEFT_WRIST, :2] - keypoints[:, RIGHT_WRIST, :2],
            axis=-1, keepdims=True)  # (seq_len, 1)
        
        # Combine all features (51 + 6 + 4 + 5 + 1 = 67 features per frame)
        features = np.concatenate([
            raw_features,         # 51
            velocity_features,    # 6
            rel_motion,           # 4
            angle_features,       # 5
            hand_dist            # 1
        ], axis=-1)
        
        return features

    def update_behavior(self, state, new_prediction, decay=0.7, time_decay=0.95):
        """Update behavior state with new prediction"""
        for behavior in state:
            state[behavior] *= time_decay
        for behavior in state:
            state[behavior] *= decay
        state[new_prediction] = (state[new_prediction] + 1) / 2
        
        total = sum(state.values())
        for behavior in state:
            state[behavior] /= total
        
        return state

    def update_gui(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

    def update_video(self):
        if self.running:
            self.after(30, self.update_video)

    def stop_camera(self):
        self.running = False
        
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
            time.sleep(0.5) 
            
        if hasattr(self, 'out') and self.out is not None:
            self.out.release()
            self.out = None
            time.sleep(0.5) 
            
        self.canvas.delete("all")
        self.clear_all_tracking()
        
        self.video_source = None

    def close_window(self):
        self.stop_camera()
        self.cleanup()

    def store_video_in_mongodb(self, video_path):
        if not os.path.exists(video_path):
            self.update_results_text(f"Video file not found: {video_path}")
            return None
        
        try:
            original_filename = os.path.basename(self.video_source) if isinstance(self.video_source, str) else "camera_recording.mp4"
            
            existing = self.db.video_metadata.find_one({'filename': original_filename})
            if existing:
                self.update_results_text(f"Video already exists in MongoDB with ID: {existing['_id']}")
                return existing['_id']
            
            with open(video_path, 'rb') as video_file:
                video_id = self.fs.put(video_file, filename=original_filename)
                
                # Store metadata
                video_metadata = {
                    'filename': original_filename,
                    'file_id': video_id,
                    'source': self.video_source,
                    'processing_date': datetime.now(),
                    'frame_count': self.frame_count,
                    'duration': self.frame_count / self.current_fps if hasattr(self, 'current_fps') and self.current_fps else None
                }
                
                # Insert into video_metadata collection
                result = self.db.video_metadata.insert_one(video_metadata)
                self.update_results_text(f"Video stored in MongoDB with ID: {result.inserted_id}")
                
                # Delete the local file after storing
                try:
                    os.remove(video_path)
                except Exception as e:
                    self.update_results_text(f"Warning: Could not delete local video file: {str(e)}")
                    
                return result.inserted_id
        except Exception as e:
            self.update_results_text(f"Error storing video in MongoDB: {str(e)}")
            return None

    def store_summary_in_mongodb(self, video_metadata_id=None):
        if not hasattr(self, 'last_summary'):
            self.update_results_text("No summary available to store")
            return None
        
        try:
            fighting_incidents = [
                {
                    'person_id': int(pid), 
                    'incidents': [
                        {
                            'start': float(start), 
                            'end': float(end)
                        }
                        for start, end in incidents
                        if end is not None and end >= start
                    ]
                }
                for pid, incidents in self.fighting_incidents.items()
            ]
            
            running_incidents = [
                {
                    'person_id': int(pid),
                    'incidents': [
                        {
                            'start': float(start),
                            'end': float(end)
                        }
                        for start, end in incidents
                        if end is not None and end >= start
                    ]
                }
                for pid, incidents in self.running_incidents.items()
            ]
            
            loitering_incidents = [
                {
                    'person_id': int(pid),
                    'incidents': [
                        {
                            'start': float(start),
                            'end': float(end)
                        }
                        for start, end in incidents
                        if end is not None
                    ]
                }
                for pid, incidents in self.loitering_incidents.items()
            ]
            
            coverage_incidents = [
                {
                    'start': float(start),
                    'end': float(end),
                    'person_id': int(pid) if pid is not None else None
                }
                for start, end, pid in self.coverage_incidents
                
                if end is not None
            ]
            
            summary_data = {
                'timestamp': datetime.now(),
                'video_metadata_id': video_metadata_id,
                'summary_text': self.last_summary,
                'fighting_incidents': fighting_incidents,
                'running_incidents': running_incidents,
                'loitering_incidents': loitering_incidents,
                'coverage_incidents': coverage_incidents
            }
            
            result = self.db.behavior_incidents.insert_one(summary_data)
            self.update_results_text(f"Behavior summary stored in MongoDB with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            self.update_results_text(f"Error storing summary in MongoDB: {str(e)}")
            return None

    def store_analysis_in_mongodb(self, summary_id=None, video_metadata_id=None):
        pass

    def play_alert_sound(self, behavior_type):
        if not self.is_camera_mode or self.alert_sound is None:
            return
            
        current_time = time.time()
        
        if not self.alert_playing[behavior_type] and current_time - self.last_alert_time[behavior_type] >= self.alert_cooldown:
            try:
                if self.alert_channel[behavior_type] is None or not self.alert_channel[behavior_type].get_busy():
                    self.alert_channel[behavior_type] = self.alert_sound.play(loops=-1)  
                self.alert_playing[behavior_type] = True
                self.last_alert_time[behavior_type] = current_time
            except Exception as e:
                self.update_results_text(f"Error playing alert sound: {str(e)}")

    def stop_alert_sound(self, behavior_type):
        if self.alert_channel[behavior_type] is not None:
            try:
                self.alert_channel[behavior_type].stop()
                self.alert_playing[behavior_type] = False
            except Exception as e:
                self.update_results_text(f"Error stopping alert sound: {str(e)}")

    def cleanup(self):
        try:
            # Stop all alert sounds
            for behavior_type in self.alert_channel:
                if self.alert_channel[behavior_type] is not None:
                    self.alert_channel[behavior_type].stop()
            pygame.mixer.quit()
        except:
            pass
        self.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    app = BehaviorDetectionWindow(root)
    app.mainloop()


