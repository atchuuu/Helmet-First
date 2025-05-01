import cv2
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# Load YOLOv5 model with trained weights
weights = "../runs/train/exp/weights/best.pt"  # Relative path from yolov5 directory
device = select_device('cpu')
model = DetectMultiBackend(weights, device=device, dnn=False)
model.eval()

class HelmetDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Helmet Detection")
        self.root.geometry("400x300")

        # Create buttons
        tk.Button(root, text="Upload Picture", command=self.upload_picture).pack(pady=10)
        tk.Button(root, text="Upload Video", command=self.upload_video).pack(pady=10)
        tk.Button(root, text="Live Webcam", command=self.live_webcam).pack(pady=10)
        tk.Button(root, text="Exit", command=root.quit).pack(pady=10)

        # Variable for video/webcam display
        self.cap = None
        self.is_running = False

    def process_frame(self, frame):
        # Preprocess image
        orig_shape = frame.shape[:2]  # Original height, width
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))  # Match training size
        img = torch.from_numpy(img).to(device).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        # YOLOv5 prediction with adjusted confidence threshold
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.4)[0]  # Lowered thresholds for more detections

        if pred is not None and len(pred):
            for *box, conf, cls in pred:
                # Override label based on confidence threshold of 0.85
                label = "with_helmet" if conf > 0.85 else "without_helmet"
                color = (0, 255, 0) if conf > 0.85 else (0, 0, 255)  # Green for > 0.85, Red for <= 0.85
                # Convert box to tensor for scaling
                box_tensor = torch.tensor(box[:4], dtype=torch.float32).unsqueeze(0).to(device)
                # Scale boxes back to original image size
                scaled_box = scale_boxes((640, 640), box_tensor, orig_shape)[0]
                xmin, ymin, xmax, ymax = map(int, scaled_box)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def upload_picture(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path and os.path.exists(file_path):
            image = cv2.imread(file_path)
            if image is not None:
                processed_image = self.process_frame(image)
                cv2.imshow("Helmet Detection", processed_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                messagebox.showerror("Error", "Failed to load image")
        else:
            messagebox.showerror("Error", "File not found")

    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file_path and os.path.exists(file_path):
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                self.is_running = True
                while self.is_running:
                    ret, frame = cap.read()
                    if not ret:
                        messagebox.showinfo("Info", "End of video")
                        break
                    processed_frame = self.process_frame(frame)
                    cv2.imshow("Helmet Detection", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.is_running = False
                cap.release()
                cv2.destroyAllWindows()
            else:
                messagebox.showerror("Error", "Failed to open video")
        else:
            messagebox.showerror("Error", "File not found")

    def live_webcam(self):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            self.is_running = True
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    messagebox.showerror("Error", "Failed to grab frame")
                    break
                processed_frame = self.process_frame(frame)
                cv2.imshow("Helmet Detection", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
            cap.release()
            cv2.destroyAllWindows()
        else:
            messagebox.showerror("Error", "Failed to open webcam")

if __name__ == "__main__":
    root = tk.Tk()
    app = HelmetDetectionApp(root)
    root.mainloop()