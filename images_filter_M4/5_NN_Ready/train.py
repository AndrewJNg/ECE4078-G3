import torch
from pathlib import Path
import os

def train_yolov8(data_yaml, cfg_yaml, weights=None, batch_size=16, max_epochs=100, patience=10, device='0', output_dir='runs/train'):

    # Set up device
    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

    # Initialize variables for early stopping
    best_metric = float('inf')  # Initialize with a high value (for loss, use 'inf'; for mAP, use 0)
    no_improvement_count = 0

    # Run YOLOv8 train script
    for epoch in range(max_epochs):
        train_command = f'python train.py --batch-size {batch_size} --epochs 1 --data {data_yaml} --cfg {cfg_yaml} --device {device} --exist-ok --project {output_dir} --epoch {epoch+1}'
        
        if weights is not None:
            train_command += f' --weights {weights}'

        os.system(train_command)

        # Parse the loss from the log file
        log_file = os.path.join(output_dir, 'train.log')
        loss = None
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if 'loss' in line:
                    loss = float(line.split(' ')[-1].strip())
                    break

        if loss is not None:
            print(f'Epoch {epoch+1}/{max_epochs}, Loss: {loss:.4f}')

        # Check validation metric (e.g., validation loss) to decide whether to stop early
        # Replace this with the actual metric you want to monitor (e.g., mAP)
        current_metric = 0.0  # Replace with your validation metric value for the current epoch

        if current_metric < best_metric:
            best_metric = current_metric
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f'Early stopping: No improvement for {patience} consecutive epochs.')
            break

if __name__ == '__main__':
    data_yaml = 'data.yaml'  # Path to your data.yaml file
    cfg_yaml = 'models/yolov8.yaml'  # Path to your model configuration file (e.g., yolov8.yaml)
    weights = None  # Path to pre-trained weights (optional, set to None for training from scratch)
    batch_size = 16  # Adjust this based on your hardware
    max_epochs = 100  # Maximum number of training epochs
    patience = 10  # Number of consecutive epochs with no improvement to trigger early stopping
    device = '0'  # GPU device index (e.g., '0' for the first GPU)
    output_dir = 'runs/train'  # Directory to save training results

    train_yolov8(data_yaml, cfg_yaml, weights, batch_size, max_epochs, patience, device, output_dir)