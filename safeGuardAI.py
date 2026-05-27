import ttkbootstrap as tb
import tkinter as tk
from tkinter import font as tkfont, Button, filedialog
from ttkbootstrap.constants import *  
import cv2
from PIL import Image, ImageTk
import threading
import time
import numpy as np
from ultralytics import YOLO
import os
from collections import defaultdict
import json
import requests
from datetime import datetime
import re
from pymongo import MongoClient
from gridfs import GridFS
import pygame
from tkinter import ttk

class ObjectDetectionWindow(tb.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Suspicious Object Detection")
        self.state("zoomed")
        self.configure(bg="#e8f5e9")  
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Load alert sound
        self.alert_sound = None
        try:
            sound_path = os.path.join(os.path.dirname(__file__), "security_alarm.mp3")
            self.alert_sound = pygame.mixer.Sound(sound_path)
        except Exception as e:
            print(f"Error loading alert sound: {e}")
        
        # Alert settings
        self.last_alert_time = defaultdict(float)  # Track last alert time for each type
        self.alert_cooldown = 3.0    # Minimum seconds between alerts
        self.alert_playing = defaultdict(bool)  # Track if alert is currently playing
        self.alert_channel = defaultdict(lambda: None)  # Track sound channels
        self.is_camera_mode = False  # Flag for camera mode

        # MongoDB Connection
        try:
            self.mongo_client = MongoClient('mongodb://localhost:27017/')
            self.db = self.mongo_client['suspiciousObjectDatabase']
            self.fs = GridFS(self.db)
            
            if 'object_incidents' not in self.db.list_collection_names():
                self.db.create_collection('object_incidents')
            if 'video_metadata' not in self.db.list_collection_names():
                self.db.create_collection('video_metadata')
            if 'analysis_results' not in self.db.list_collection_names():
                self.db.create_collection('analysis_results')
                
            print("Connected to MongoDB successfully")
        except Exception as e:
            print(f"Error connecting to MongoDB: {str(e)}")
     
        self.paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        self.left_frame = tb.Frame(self.paned_window, style='light.TFrame')
        self.paned_window.add(self.left_frame, weight=1)

        self.right_frame = tb.Frame(self.paned_window, style='light.TFrame')
        self.paned_window.add(self.right_frame, weight=1)

        self.update() 
        window_width = self.winfo_width()
        self.paned_window.sashpos(0, window_width // 2)

        self.setup_left_frame()
        self.setup_right_frame()

        # Initialize video capture
        self.cap = None
        self.running = False
        self.detection_thread = None
        self.out = None 

        # Load YOLO models
        self.model = YOLO('yolov8n.pt')
        self.knifeModel = YOLO("runs/detect/train/weights/best.pt")

        # Define class indices
        self.PERSON_CLASS = 0
        self.BAG_CLASSES = [24, 26, 28]

        # Tracking data
        self.object_tracks = defaultdict(list)
        self.unattended_objects = set()
        self.unattended_bboxes = {}
        self.unattended_times = {}

        # Detection events
        self.detection_events = []

        # Thresholds
        self.DIST_THRESHOLD = 50
        self.UNATTENDED_TIME_THRESHOLD = 5

        # Output folder
        self.output_folder = "objectOutputVideo"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.paned_window.paneconfig(self.left_frame, stretch='always')
        self.paned_window.paneconfig(self.right_frame, stretch='always')

        style = tb.Style()
        style.configure('light.TFrame', background='#e8f5e9')

    def setup_left_frame(self):
        label = tb.Label(
            self.left_frame, 
            text="Suspicious Object Detection",
            font=("Helvetica", 16),
            bootstyle="primary",
            background="#e8f5e9",
            foreground="#004d40"
        )
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
        self.upload_button.pack(side=tk.LEFT, padx=5)

        self.camera_button = tb.Button(
            self.button_frame,
            text="Open Camera",
            command=self.open_camera,
            bootstyle="success",
            width=15
        )
        self.camera_button.pack(side=tk.LEFT, padx=5)

        self.close_camera_button = tb.Button(
            self.button_frame,
            text="Close Camera",
            command=self.close_camera,
            bootstyle="danger",
            width=15
        )
        self.close_camera_button.pack(side=tk.LEFT, padx=5)
        self.close_camera_button.pack_forget()

        self.end_video_button = tb.Button(
            self.button_frame,
            text="End Video",
            command=self.end_video,
            bootstyle="danger",
            width=15
        )
        self.end_video_button.pack(side=tk.LEFT, padx=5)
        self.end_video_button.pack_forget()

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

    def play_alert_sound(self, alert_type):
        """Play alert sound continuously while threat is detected"""
        if not self.is_camera_mode or self.alert_sound is None:
            return
            
        current_time = time.time()
        
        if not self.alert_playing[alert_type] and current_time - self.last_alert_time[alert_type] >= self.alert_cooldown:
            try:
                if self.alert_channel[alert_type] is None or not self.alert_channel[alert_type].get_busy():
                    self.alert_channel[alert_type] = self.alert_sound.play(loops=-1)  
                self.alert_playing[alert_type] = True
                self.last_alert_time[alert_type] = current_time
            except Exception as e:
                print(f"Error playing alert sound: {e}")

    def stop_alert_sound(self, alert_type):
        #Stop the alert sound
        if self.alert_channel[alert_type] is not None:
            try:
                self.alert_channel[alert_type].stop()
                self.alert_playing[alert_type] = False
            except Exception as e:
                print(f"Error stopping alert sound: {e}")

    def cleanup(self):
        #Clean up pygame resources
        try:
            for alert_type in self.alert_channel:
                if self.alert_channel[alert_type] is not None:
                    self.alert_channel[alert_type].stop()
            pygame.mixer.quit()
        except:
            pass
        self.destroy()

    def reset_detection_data(self):
        #Reset all detection and tracking data
        self.object_tracks.clear()
        self.unattended_objects.clear()
        self.unattended_bboxes.clear()
        self.unattended_times.clear()
        self.detection_events.clear()
        self.alert_playing.clear()
        self.alert_channel.clear()
        self.last_alert_time.clear()

    def upload_video(self):
        self.is_camera_mode = False 
        self.stop_camera()
        self.clear_results()
        self.reset_detection_data() 

        self.grab_set()  

        file_path = filedialog.askopenfilename(
            parent=self, 
            filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]
        )

        self.grab_release()

        if file_path:
            self.video_source = file_path
            self.start_video()
            self.end_video_button.pack(side=tk.LEFT, padx=10)
            self.close_camera_button.pack_forget()
            self.upload_button.pack_forget()
            self.camera_button.pack_forget()
            self.api_button.pack_forget() 

    def open_camera(self):
        self.is_camera_mode = True  
        self.stop_camera()
        self.clear_results()
        self.reset_detection_data()  
        self.video_source = 0
        self.start_video()
        self.close_camera_button.pack(side=tk.LEFT, padx=10)
        self.end_video_button.pack_forget()
        self.upload_button.pack_forget()
        self.camera_button.pack_forget()
        self.api_button.pack_forget()  

    def close_camera(self):
        for alert_type in list(self.alert_playing.keys()):
            self.stop_alert_sound(alert_type)
            
        if hasattr(self, 'output_folder') and os.path.exists(self.output_folder):
            video_files = [f for f in os.listdir(self.output_folder) if f.endswith('.mp4')]
            if video_files and not hasattr(self, '_video_stored'):
                latest_video = os.path.join(self.output_folder, video_files[-1])
                video_metadata_id = self.store_video_in_mongodb(latest_video)
                self._video_stored = True  
                
                summary_id = self.store_summary_in_mongodb(video_metadata_id)
        
        self.stop_camera()
        self.close_camera_button.pack_forget()
        self.upload_button.pack(side=tk.LEFT, padx=10)
        self.camera_button.pack(side=tk.LEFT, padx=10)
        self.api_button.pack(pady=10)
        self.display_detection_results()
        print("Camera closed - Data stored in MongoDB")

    def end_video(self):
        if hasattr(self, 'output_folder') and os.path.exists(self.output_folder):
            video_files = [f for f in os.listdir(self.output_folder) if f.endswith('.mp4')]
            if video_files and not hasattr(self, '_video_stored'):
                latest_video = os.path.join(self.output_folder, video_files[-1])
                video_metadata_id = self.store_video_in_mongodb(latest_video)
                self._video_stored = True 
                
                summary_id = self.store_summary_in_mongodb(video_metadata_id)
        
        self.stop_camera()
        self.end_video_button.pack_forget()
        self.upload_button.pack(side=tk.LEFT, padx=10)
        self.camera_button.pack(side=tk.LEFT, padx=10)
        self.api_button.pack_forget()
        self.api_button.pack(pady=10)
        self.display_detection_results()
        print("Video processing ended - Data stored in MongoDB")

    def clear_results(self):
        self.results_text.delete(1.0, tk.END)
        self.detection_events.clear()
        
    def remove_markdown(self, text):
        text = re.sub(r'#\s*', '', text)           
        text = re.sub(r'\*{1,2}', '', text)        
        text = re.sub(r'`+', '', text)             
        text = re.sub(r'>\s*', '', text)           
        text = re.sub(r'\n\s*-\s*', '\n', text)    
        return text
    
    def update_results_text(self, event_type, start_time, confidence=None):
        start_time_str = self.format_time(start_time)
        confidence_str = f" (Confidence: {confidence:.2f})" if confidence is not None else ""
        result_line = f"{event_type} detected at {start_time_str}{confidence_str}\n"
        
        self.results_text.insert(tk.END, result_line)
        
        self.results_text.see(tk.END)

    def display_detection_results(self):
        self.results_text.delete(1.0, tk.END)

        self.results_text.insert(tk.END, "Detection Events:\n\n")

        for event in self.detection_events:
            event_type, start_time, end_time, confidence = event
            start_time_str = self.format_time(start_time)
            end_time_str = self.format_time(end_time) if end_time is not None else "Ongoing"
            confidence_str = f" (Avg Confidence: {confidence:.2f})" if confidence is not None else ""
            result_line = f"{event_type} detected at {start_time_str} - Ended at {end_time_str}{confidence_str}\n"
            self.results_text.insert(tk.END, result_line)

        self.results_text.see(tk.END)

    def start_video(self):
        if self.cap:
            self.running = False
            self.cap.release()

        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print("Error: Cannot open video source.")
            return

        self.running = True
        self.detection_thread = threading.Thread(target=self.capture_and_process_frames, daemon=True)
        self.detection_thread.start()

        self.update_video()

    def send_to_api(self):
        try:
            # Get the latest video_metadata_id and summary_id from MongoDB
            latest_video = self.db.video_metadata.find_one(sort=[('processing_date', -1)])
            latest_summary = self.db.object_incidents.find_one(sort=[('timestamp', -1)])
            
            video_metadata_id = latest_video['_id'] if latest_video else None
            summary_id = latest_summary['_id'] if latest_summary else None

            prompt = f"""Please analyze this object detection summary. Here are the detected events:

{self.format_detection_events()}

Please provide:
1. A detailed analysis of all detected objects and events
2. Analysis of any patterns in object detection (e.g., frequent unattended bags)
3. Security recommendations based on the detected objects
4. Risk assessment considering the frequency and severity of detections

Important: Focus on analyzing all incidents shown in the summary above."""

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
                        "content": "You are a security analysis expert. Your task is to analyze the object detection data and provide insights about security risks."
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
                analysis = response.json()
                if analysis and "choices" in analysis and len(analysis["choices"]) > 0:
                    analysis_text = analysis["choices"][0]["message"]["content"]
                    clean_text = self.remove_markdown(analysis_text)
                    
                    # Clear and update results text
                    self.results_text.delete(1.0, tk.END)
                    self.results_text.insert(tk.END, "\nAnalysis Results:\n")
                    self.results_text.insert(tk.END, clean_text)
                    
                    # Store analysis in MongoDB
                    self.store_analysis_in_mongodb(clean_text, summary_id, video_metadata_id)
                else:
                    self.results_text.insert(tk.END, "\nNo analysis results in API response")
            else:
                self.results_text.insert(tk.END, f"\nError sending to API: {response.status_code}")
                self.results_text.insert(tk.END, response.text)

        except Exception as e:
            self.results_text.insert(tk.END, f"\nError preparing/sending data: {str(e)}")

    def format_detection_events(self):
        formatted_text = "Detection Events:\n\n"
        for event in self.detection_events:
            event_type, start_time, end_time, confidence = event
            start_time_str = self.format_time(start_time)
            end_time_str = self.format_time(end_time) if end_time is not None else "Ongoing"
            confidence_str = f" (Confidence: {confidence:.2f})" if confidence is not None else ""
            formatted_text += f"- {event_type} detected at {start_time_str} - Ended at {end_time_str}{confidence_str}\n"
        
        if self.unattended_objects:
            formatted_text += "\nDetected Object Types:\n"
            object_type_names = {
                24: "Handbag",
                26: "Backpack",
                28: "Suitcase"
            }
            for obj_id in self.unattended_objects:
                obj_type = object_type_names.get(obj_id, f"Unknown bag type (ID: {obj_id})")
                formatted_text += f"- {obj_type}\n"
        
        return formatted_text

    def capture_and_process_frames(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        video_name = os.path.basename(self.video_source) if self.video_source else "camera_output.mp4"
        output_video_path = os.path.join(self.output_folder, video_name)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_video_path, fourcc, fps, (640, 480))

        frame_counter = 0
        last_processed_frame = None

        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                frame_counter += 1

                # Calculate video time
                video_time = frame_counter / fps

                if frame_counter % 3 == 0:
                    results = self.model(frame, conf=0.5)
                    knifeResults = self.knifeModel(frame, conf=0.5)
                    UnattendedResults = self.model(frame)

                    detections = [box.xyxy[0].tolist() for box in results[0].boxes if int(box.cls) == 0]
                    knife_detections = knifeResults[0]

                    last_processed_frame = self.suspicious_detection(frame.copy(), detections, knife_detections, UnattendedResults, video_time)

                if last_processed_frame is not None:
                    display_frame = last_processed_frame
                else:
                    display_frame = frame

                self.out.write(display_frame)

                self.update_gui(display_frame)

            else:
                break  

        self.cap.release()
        if self.out:
            self.out.release()
            self.out = None

        self.display_detection_results()

        print("\nDetection Events:")
        for event in self.detection_events:
            event_type, start_time, end_time, confidence = event
            start_time_str = self.format_time(start_time)
            end_time_str = self.format_time(end_time) if end_time is not None else "Ongoing"
            confidence_str = f" (Avg Confidence: {confidence:.2f})" if confidence is not None else ""
            print(f"{event_type} detected at {start_time_str} - Ended at {end_time_str}{confidence_str}")

        cv2.destroyAllWindows()
        for i in range(5): 
            cv2.waitKey(1)

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
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.out:
            self.out.release()
            self.out = None
        self.canvas.delete("all")
        # Redraw black rectangle as default display
        self.canvas.create_rectangle(0, 0, 640, 480, fill="black", outline="gray", width=2)

    def close_window(self):
        self.stop_camera()
        self.destroy()

    def format_time(self, seconds):
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}:{secs:02d}"

    def store_video_in_mongodb(self, video_path):
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return None
        
        try:
            with open(video_path, 'rb') as video_file:
                # Get video information
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                # Use the original source video name
                original_filename = os.path.basename(self.video_source) if isinstance(self.video_source, str) else "camera_recording.mp4"
                
                # Always store a new copy in GridFS
                video_id = self.fs.put(video_file, filename=original_filename)
                
                # Store metadata about the video
                video_metadata = {
                    'filename': original_filename,
                    'file_id': video_id,
                    'source': str(self.video_source), 
                    'processing_date': datetime.now(),
                    'frame_count': frame_count,
                    'duration': duration,
                    'fps': fps
                }
                
                # Insert new metadata entry
                result = self.db.video_metadata.insert_one(video_metadata)
                print(f"Video stored in MongoDB with ID: {result.inserted_id}")
                print(f"Video metadata: {video_metadata}")  # Debug print
                return result.inserted_id
        except Exception as e:
            print(f"Error storing video in MongoDB: {str(e)}")
            return None

    def store_summary_in_mongodb(self, video_metadata_id=None):
        """Store object detection summary in MongoDB"""
        try:
            # Convert detection events to proper format
            formatted_events = []
            active_unattended_count = 0  # Count of currently active unattended objects
            
            for event in self.detection_events:
                event_type, start_time, end_time, confidence = event
                formatted_event = {
                    'event_type': event_type,
                    'start_time': float(start_time),
                    'end_time': float(end_time) if end_time is not None else None,
                    'confidence': float(confidence) if confidence is not None else None
                }
                formatted_events.append(formatted_event)
                
                # Count unattended object events that haven't ended
                if event_type == "Unattended Object" and end_time is None:
                    active_unattended_count += 1

            summary_data = {
                'timestamp': datetime.now(),
                'video_metadata_id': video_metadata_id,
                'detection_events': formatted_events,
                'unattended_objects': active_unattended_count,  # Only count active unattended objects
                'total_events': len(self.detection_events)
            }
            
            # Store the summary
            result = self.db.object_incidents.insert_one(summary_data)
            print(f"Detection summary stored in MongoDB with ID: {result.inserted_id}")
            print(f"Active unattended objects: {active_unattended_count}") 
            return result.inserted_id
        except Exception as e:
            print(f"Error storing summary in MongoDB: {str(e)}")
            return None

    def store_analysis_in_mongodb(self, analysis_text, summary_id=None, video_metadata_id=None):
        #Store analysis results in MongoDB
        try:
            analysis_data = {
                'timestamp': datetime.now(),
                'summary_id': summary_id,
                'video_metadata_id': video_metadata_id,
                'analysis_text': analysis_text,
                'source': 'Groq API',
                'model': 'llama-3.3-70b-versatile'
            }
            
            result = self.db.analysis_results.insert_one(analysis_data)
            print(f"Analysis results stored in MongoDB with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            print(f"Error storing analysis in MongoDB: {str(e)}")
            return None

    def suspicious_detection(self, frame, detections, knife_detections, unattendedResults, video_time):
        detected_objects = []
        person_positions = set()
        person_bboxes = []
        knife_detected = False
        unattended_detected = False
        best_knife_box = None
        best_knife_conf = 0.0
        
        # Define object type names
        object_type_names = {
            24: "Handbag",
            26: "Backpack",
            28: "Suitcase"
        }

        # Process detections (People & Bags)
        for result in unattendedResults:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                obj_id = int(cls)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)  # Object center

                if obj_id == self.PERSON_CLASS:
                    person_positions.add((cx, cy))  # Store person positions (x, y)
                    person_bboxes.append((x1, y1, x2, y2))  # Store full bounding box
                elif obj_id in self.BAG_CLASSES:
                    detected_objects.append((cx, cy, obj_id, x1, y1, x2, y2))

        current_time = time.time()  # Get the current time

        # Track objects and check for movement
        for cx, cy, obj_id, x1, y1, x2, y2 in detected_objects:
            self.object_tracks[obj_id].append((cx, cy, len(self.object_tracks[obj_id])))

            # Check movement
            if len(self.object_tracks[obj_id]) > 1:
                prev_cx, prev_cy, _ = self.object_tracks[obj_id][-2]
                distance = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)

                if distance > self.DIST_THRESHOLD:  # Object moved
                    self.unattended_objects.discard(obj_id)
                    self.unattended_bboxes.pop(obj_id, None)
                    self.unattended_times.pop(obj_id, None)
                    continue  # Skip further processing for this object

            # Check if object is abandoned (no person nearby)
            is_abandoned = all(
                np.sqrt((cx - px) ** 2 + (cy - py) ** 2) > 80  # Euclidean distance
                for px, py in person_positions
            )

            # If object is abandoned and not already in the unattended set, start the timer
            if is_abandoned and obj_id not in self.unattended_objects:
                if obj_id not in self.unattended_times:
                    self.unattended_times[obj_id] = current_time  # Start the timer

                # If it's been unattended for long enough, mark it as unattended
                if (current_time - self.unattended_times[obj_id]) >= self.UNATTENDED_TIME_THRESHOLD:
                    self.unattended_objects.add(obj_id)
                    self.unattended_bboxes[obj_id] = (x1, y1, x2, y2)
                    unattended_detected = True
                    # Log unattended object detection
                    self.detection_events.append(("Unattended Object", video_time, None, None))
                    # Update the results text in real-time
                    self.update_results_text("Unattended Object", video_time)

        # Remove unattended objects if they overlap with a person
        for obj_id, (x1, y1, x2, y2) in list(self.unattended_bboxes.items()):
            for (px1, py1, px2, py2) in person_bboxes:
                if x1 < px2 and x2 > px1 and y1 < py2 and y2 > py1:  # Bounding box overlap
                    self.unattended_objects.discard(obj_id)
                    self.unattended_bboxes.pop(obj_id, None)
                    self.unattended_times.pop(obj_id, None)
                    # Log unattended object end time
                    for i, event in enumerate(self.detection_events):
                        if event[0] == "Unattended Object" and event[2] is None:
                            self.detection_events[i] = (event[0], event[1], video_time, None)
                    break  # Stop checking once removed

        # Draw unattended objects
        for obj_id, (x1, y1, x2, y2) in self.unattended_bboxes.items():
            elapsed_time = current_time - self.unattended_times.get(obj_id, 0)
            if elapsed_time >= self.UNATTENDED_TIME_THRESHOLD:
                unattended_detected = True
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f"Unattended Object ({int(elapsed_time)}s)", 
                            (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Handle unattended object alert
        if self.is_camera_mode:
            if unattended_detected:
                self.play_alert_sound("unattended")
            elif self.alert_playing.get("unattended", False):
                self.stop_alert_sound("unattended")

        # Detect knife
        knife_confidences = []  # To store confidence levels for the current detection
        for box in knife_detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            label = self.knifeModel.names.get(cls, "Unknown")

            if label.lower() == "knife" and conf > best_knife_conf:
                best_knife_box = (x1, y1, x2, y2)
                best_knife_conf = conf

            if conf > 0.65:
                knife_detected = True
                knife_confidences.append(conf)  # Store confidence level

        # Draw the most confident knife detection only if confidence >= 0.7
        if knife_detected and best_knife_box is not None and best_knife_conf >= 0.7:
            x1, y1, x2, y2 = best_knife_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Knife {best_knife_conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # Log knife detection
            if not any(event[0] == "Knife" and event[2] is None for event in self.detection_events):
                print(f"Logging knife detection at {video_time} with confidence {best_knife_conf:.2f}")  # Debug print
                self.detection_events.append(("Knife", video_time, None, knife_confidences))
                # Update the results text in real-time
                self.update_results_text("Knife", video_time, best_knife_conf)
            
            # Play knife alert sound if in camera mode
            if self.is_camera_mode:
                self.play_alert_sound("knife")
        else:
            # Log knife end time if knife is no longer detected
            for i, event in enumerate(self.detection_events):
                if event[0] == "Knife" and event[2] is None:
                    # Calculate average confidence
                    avg_confidence = np.mean(event[3]) if event[3] else 0.0
                    print(f"Logging knife end time at {video_time} with avg confidence {avg_confidence:.2f}")  # Debug print
                    self.detection_events[i] = (event[0], event[1], video_time, avg_confidence)
            
            # Stop knife alert sound if it was playing
            if self.is_camera_mode and self.alert_playing.get("knife", False):
                self.stop_alert_sound("knife")

        return frame


class SafeGuardAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SafeGuard AI")
        self.root.state("zoomed")

        # Set up custom fonts
        self.title_font = ("Helvetica", 36, "bold")  
        self.header_font = ("Helvetica", 18, "bold")
        self.text_font = ("Helvetica", 12)

        # Create the main container with dark green background
        self.main_container = tb.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        self.main_container.configure(style='light.TFrame')
        
        # Configure light green background
        style = tb.Style()
        style.configure('light.TFrame', background='#e8f5e9')  # Very light green color
        style.configure('transparent.TFrame', background='#e8f5e9')  # Match main background
        style.configure('transparent.TLabelframe', background='#e8f5e9')  # For the labelframes
        style.configure('transparent.TLabelframe.Label', background='#e8f5e9')  # For the labelframe labels
        
        # Create top section with dark green background
        self.top_section = tb.Frame(self.main_container)
        self.top_section.pack(fill=tk.X)
        self.top_section.configure(style='dark.TFrame')
        # Configure dark green background
        style.configure('dark.TFrame', background='#004d40')  # Dark green color

        # Add header with white text
        self.header = tb.Label(
            self.top_section,
            text="SafeGuard AI",
            font=self.title_font,
            foreground='white',
            background='#004d40',  # Match frame background
            justify=tk.LEFT  # Align text to left
        )
        self.header.pack(pady=20, padx=50, anchor='w')  # anchor='w' aligns to left

        # Add introduction text with white text
        self.intro_text = tb.Label(
            self.top_section,
            text="SafeGuard AI system is a system that can detect abnormal behavior of persons in public and alert the authorities when suspicious behavior is detected.",
            font=self.text_font,
            foreground='white',
            background='#004d40',  # Match frame background
            wraplength=1200,  # Increased wraplength to allow one line
            justify=tk.LEFT  # Align text to left
        )
        self.intro_text.pack(pady=20, padx=50, anchor='w')  # anchor='w' aligns to left

        # Add container for buttons
        self.button_container = tb.Frame(self.main_container, style='transparent.TFrame')
        self.button_container.pack(pady=(100, 20))

        # Create left and right frames for the boxes
        left_box_frame = tb.Frame(self.button_container, style='transparent.TFrame')
        left_box_frame.pack(side=tk.LEFT, padx=20)
        
        right_box_frame = tb.Frame(self.button_container, style='transparent.TFrame')
        right_box_frame.pack(side=tk.LEFT, padx=20)

        # Suspicious Object Detection Section (Left)
        self.object_box = tb.Labelframe(
            left_box_frame,
            text="",
            padding=20,
            bootstyle="primary",
            style='transparent.TLabelframe'
        )
        self.object_box.pack(fill=tk.BOTH, expand=True)

        # Create canvas for background image with moderate size
        self.object_canvas = tk.Canvas(
            self.object_box,
            width=450,  # Moderate increase from original 400
            height=250,  # Moderate increase from original 200
            highlightthickness=0
        )
        self.object_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Load and display background image
        try:
            object_bg = Image.open("images/object.png")
            object_bg = object_bg.resize((450, 250), Image.Resampling.LANCZOS)  # Adjusted size
            self.object_bg_image = ImageTk.PhotoImage(object_bg)
            self.object_canvas.create_image(0, 0, anchor=tk.NW, image=self.object_bg_image)
            
            # Create title text - adjusted position
            self.object_canvas.create_text(
                225, 40,  # Centered horizontally, adjusted vertically
                text="Suspicious Object Detection",
                font=("Helvetica", 16, "bold"),
                fill='white',
                justify=tk.CENTER
            )
            
            # Create description text - adjusted position
            self.object_canvas.create_text(
                225, 115,  # Centered horizontally, adjusted vertically
                text="Upload a video or use the camera for real-time object detection.",
                font=self.text_font,
                fill='white',
                width=380,
                justify=tk.CENTER
            )
        except Exception as e:
            print(f"Error loading object detection background: {e}")

        # Create a style for the buttons
        style = tb.Style()
        style.configure('success.TButton', borderwidth=0, relief="flat", borderradius=20)

        self.object_button = tb.Button(
            self.object_canvas,
            text="Get Started",
            command=self.open_object_detection,
            bootstyle="success",
            width=15,
            style='success.TButton'
        )
        self.object_canvas.create_window(225, 175, window=self.object_button)  # Centered horizontally, adjusted vertically

        # Suspicious Behavior Detection Section (Right)
        self.behavior_box = tb.Labelframe(
            right_box_frame,
            text="",
            padding=20,
            bootstyle="primary",
            style='transparent.TLabelframe'
        )
        self.behavior_box.pack(fill=tk.BOTH, expand=True)

        # Create canvas for background image with moderate size
        self.behavior_canvas = tk.Canvas(
            self.behavior_box,
            width=450,  # Moderate increase from original 400
            height=250,  # Moderate increase from original 200
            highlightthickness=0
        )
        self.behavior_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Load and display background image
        try:
            behavior_bg = Image.open("images/behavior.png")
            behavior_bg = behavior_bg.resize((450, 250), Image.Resampling.LANCZOS)  # Adjusted size
            self.behavior_bg_image = ImageTk.PhotoImage(behavior_bg)
            self.behavior_canvas.create_image(0, 0, anchor=tk.NW, image=self.behavior_bg_image)
            
            # Create title text - adjusted position
            self.behavior_canvas.create_text(
                225, 40,  # Centered horizontally, adjusted vertically
                text="Suspicious Behavior Detection",
                font=("Helvetica", 16, "bold"),
                fill='white',
                justify=tk.CENTER
            )
            
            # Create description text - adjusted position
            self.behavior_canvas.create_text(
                225, 115,  # Centered horizontally, adjusted vertically
                text="Upload a video or use the camera for real-time behavior detection.",
                font=self.text_font,
                fill='white',
                width=380,
                justify=tk.CENTER
            )
        except Exception as e:
            print(f"Error loading behavior detection background: {e}")

        # Create a style for the buttons
        style = tb.Style()
        style.configure('success.TButton', borderwidth=0, relief="flat", borderradius=20)

        self.behavior_button = tb.Button(
            self.behavior_canvas,
            text="Get Started",
            command=self.open_behavior_detection,
            bootstyle="success",
            width=15,
            style='success.TButton'
        )
        self.behavior_canvas.create_window(225, 175, window=self.behavior_button)  # Centered horizontally, adjusted vertically

    def open_object_detection(self):
        obj_window = ObjectDetectionWindow(self.root)
        obj_window.lift() 

    def open_behavior_detection(self):
        from behaviorWindow import BehaviorDetectionWindow 
        bhv_window=BehaviorDetectionWindow(self.root)
        bhv_window.lift()


if __name__ == "__main__":
    root = tk.Tk()
    app = SafeGuardAIApp(root)
    root.mainloop()


