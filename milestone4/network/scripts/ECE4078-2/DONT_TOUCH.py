from ultralytics import YOLO
from roboflow import Roboflow
import os
import torch
import multiprocessing
3
def train_yolo():
    HOME = os.getcwd()
    print(HOME)

    # Replace "your_api_key_here" with your actual Roboflow API key
    rf = Roboflow(api_key="RQ53DJnBpntesi6vrmNd")
    project = rf.workspace("andrewjng").project("ece4078-oqvae")
    dataset = project.version(2).download("yolov8")

    # Initialize YOLO model
    model = YOLO('yolov8n.pt')

    # Train the model
    results = model.train(data=f'D:\\Monash\\Y3S1\\ECE4078\\ECE4078-G3\\milestone4\\network\\scripts\\ECE4078-2\\data.yaml', epochs=100, imgsz=640, batch=8, plots=True, device='0', patience=70)


if __name__ == '__main__':
    # Ensure that multiprocessing is used only in the main module
    multiprocessing.set_start_method('spawn', force=True)
    
    # Start the training process
    train_yolo()