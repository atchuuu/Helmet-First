# Helmet Detection with YOLOv5

This project uses YOLOv5 to detect helmets in images, videos, or live webcam feeds. It includes a trained model, a Tkinter-based GUI application, and a script for testing inference.

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/HelmetDetectionYOLOv5.git
   cd HelmetDetectionYOLOv5
   ```

2. **Install Git LFS to Download the Weights**:
   - Git LFS is required to download the trained model weights (`best.pt`).
   ```bash
   git lfs install
   git lfs pull
   ```

3. **Create a Virtual Environment and Install Dependencies**:
   - Ensure you have Python installed (version 3.12 recommended as per your setup).
   ```bash
   python -m venv yolov5_env
   yolov5_env\Scripts\activate  # On Windows
   # source yolov5_env/bin/activate  # On Linux/Mac
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   - Use the Tkinter-based GUI to upload images, videos, or use the webcam for helmet detection.
   ```bash
   python helmet_app.py
   ```

## Files
- `helmet_app.py`: Main application with a Tkinter GUI for helmet detection (image, video, webcam).
- `test_inference.py`: Script for testing model inference on a single image.
- `../../runs/train/exp/weights/best.pt`: Trained YOLOv5 model weights (stored using Git LFS).
- `helmet.yaml`: Configuration file for dataset and class names.
- `requirements.txt`: List of dependencies for the project.

## Project Structure
- The project is based on the YOLOv5 repository cloned into the `yolov5` directory.
- The trained model weights are located in the `runs/train/exp/weights/` directory (relative to the parent directory).
- The dataset used for training is in the `dataset/` directory (not included in this repository).

## Notes
- The model may require retraining if predictions are inaccurate (e.g., misclassifying helmet presence). Check the training data labels and `helmet.yaml` for class order.
- To retrain the model, use the following command:
  ```bash
  python train.py --img 640 --batch 8 --epochs 50 --data helmet.yaml --weights yolov5s.pt --project "../runs/train" --exist-ok
  ```

## Requirements
- Python 3.12
- Git LFS
- Dependencies listed in `requirements.txt` (e.g., `torch`, `opencv-python`, `tkinter`)