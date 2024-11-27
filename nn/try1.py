from ultralytics import YOLO
from pathlib import Path
import cv2, onnx, onnxoptimizer,numpy, time, threading
#import torch.onnx
import torchvision.models as models
from db.main import *
from pprint import pprint as pt


class Main_field2:
    def __init__(self, model, size, source, conf_):
        self.model = self.load_model(model)  # Initialize optimized model
        self.size = size
        self.source = source  # 0 for default camera, URL for video processing
        self.conf_ = conf_

    def load_model(self, model):
        model.fuse()  # Speed up model operation
        return model

    def process_video(self, video_stream):
        cap = cv2.VideoCapture(self.source)  # Access default webcam (index 0)
        while True:  # Main loop for video processing
            ret, frame = cap.read()  # Process Webcam Feed
            if not ret:
                break

            results = self.model.track(frame, persist=True, tracker='bytetrack.yaml', conf=self.conf_, imgsz=self.size)  # Model parameters
            dict_ = [result.names for result in results][0]  # Dictionary of dataset structure
            xyxys = [result.boxes.cpu().numpy().xyxy for result in results][0]  # xyxy box parameters (top-left, bottom-right on grid)
            conf = [result.boxes.cpu().numpy().conf for result in results][0]  # Confidence model levels
            cls_ids = [result.boxes.cpu().numpy().cls.astype(int).tolist() for result in results][0]  # itr->cpu->nparray->indices->list
            masks_output = [dict_[itr] for itr in cls_ids]  # List of faces by index
            db_output = [check_(local_list, str(itr)) for itr in masks_output]  # Output from database (Google Sheets)

            # Send frame and parameters to the video stream
            video_stream.update_frame(frame, results[0].plot(), db_output)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

    def start_streaming(self):
        video_stream = VideoStream()
        threading.Thread(target=self.process_video, args=(video_stream,)).start()
        return video_stream
    

    
class VideoStream:
    def __init__(self):
        
        self.frame = None
        self.plot = None
        self.db_output = None
        self.lock = threading.Lock()

    def update_frame(self, frame, plot, db_output):
        with self.lock:
            self.frame = frame
            self.plot = plot
            self.db_output = db_output

    def get_frame(self):
        with self.lock:
            return self.frame

    def get_plot(self):
        with self.lock:
            return self.plot

    def get_db_output(self):
        with self.lock:
            return self.db_output

def train_(model_path: any, data_set_path: any, epochs: int, size: int): #функция тренеровки модели
    model_path.info() # Display model information (optional)
    model_path.train(data=data_set_path, epochs=epochs, imgsz=size) # Train the model on the COCO8 example dataset for 100 epochs

def run_img_in_cd(model): #прогон модели по папке с img
    for num in Path("C:/Users/faso3/Documents/IT/Python/Detect/NN/img").rglob("*"): model(num) # Run inference with the YOLOv9c model on the 'bus.jpg' image

def init_model(path: str) -> any: return YOLO(path)

class FaceRecognition:

    def __init__(self, capture_index) -> None:
        self.capture_index=capture_index
        self.model=self.load_model()

    def load_model(self):
        model=YOLO('nn/models/yolov8s.pt')
        model.fuse()
        return model
    
    def predic(self,frame):
        results=self.model(frame)

    def plot_boxes(self, results, frame):
        xyxys=[]
        confidences=[]
        class_ids=[]
        for result in results:
            xyxys.append(result.boxes.cpu().numpy().xyxy)
            confidences.append(result.boxes.cpu().numpy().conf)
            class_ids.append(result.boxes.cpu().numpy().cls)

        return result[0].plot(),xyxys,confidences,class_ids
    
    def __call__(self):
        cap=cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,500)

def conv_pt_onnx(pt): #преобразование в ONNX формат
    pt.info()
    pt.export(format='onnx')  # creates onnx model

def opt_onnx(model): #оптимизация ONNX модели
    optimized_model = onnxoptimizer.optimize(model)
    onnx.save(optimized_model, 'onxx')



