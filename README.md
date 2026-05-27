# SafeGuard AI — Intelligent Public Safety Surveillance System

SafeGuard AI is an AI-powered real-time surveillance system designed to detect suspicious objects and analyze human behavior in public environments. The system integrates computer vision, deep learning, temporal modeling, and large language models to provide intelligent risk assessment and automated alert generation.


## Key Features

- Real-time object detection using YOLO
- Human pose estimation using YOLO-Pose
- Behavioral analysis using Temporal Convolutional Network (TCN)
- Suspicious activity detection 
- LLM-based risk interpretation and recommendation system
- Dataset logging and evaluation pipeline
- Automated alert system for potential threats


## System Architecture

1. Video Input  
   CCTV, uploaded video, or real-time camera feed

2. Object Detection (YOLO)  
   Detects weapons, bags, and suspicious objects

3. Pose Estimation (YOLO-Pose)  
   Extracts human keypoints for movement analysis

4. Temporal Modeling (TCN)  
   Analyzes behavior patterns over time

5. LLM Reasoning Layer  
   Generates risk level assessment and recommended actions

6. Output System  
   Alerts, visualization, and logging (MongoDB)


## Tech Stack

- Python
- PyTorch / TensorFlow
- YOLOv8
- YOLO-Pose
- Temporal Convolutional Networks (TCN)
- OpenCV
- MongoDB
- Large Language Model API (Groq / LLM inference)



## AI Models Used

- YOLOv8 (Object Detection)  
- YOLOv8-Pose (Human Keypoints)  
- TCN (Temporal sequence modeling)  
- LLM (Risk classification and explanation generation)  


## Example Output

- Real-time bounding boxes for detected objects  
- Human pose skeleton visualization  
- Behavior classification labels  
- Risk level (Low / Medium / High)  
- Automated safety recommendations  


## Academic Project

Final Year Project  
Bachelor of Science (Hons) Information Technology (Intelligent Systems)  
Universiti Utara Malaysia (UUM)


## Future Improvements

- Edge deployment (Jetson / Raspberry Pi)
- Multi-camera tracking system
- Improved LLM reasoning accuracy
- Real-time dashboard interface
