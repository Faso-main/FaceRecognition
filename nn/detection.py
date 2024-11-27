from ultralytics import YOLO
import cv2, numpy
from db.main import check_, local_list
from pprint import pprint as pt

class MainField:
    def __init__(self, model: YOLO, size: int, source: int, conf: float) -> None:
        self.model = self.load_model(model)
        self.size = size
        self.source = source
        self.conf = conf
        
    def load_model(self, model: YOLO) -> YOLO:
        model.fuse()  # Optimize model inference speed
        return model

    def process_video(self) -> None:
        cap = cv2.VideoCapture(self.source)

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Predict with the model
            results = self.model.predict(frame, conf=self.conf, imgsz=self.size)

            # Process the first result efficiently
            if results:  # Check if any results were returned
                result = results[0]
                boxes = result.boxes
                xyxys = boxes.xyxy.cpu().numpy()  # Bounding box coordinates
                confs = boxes.conf.cpu().numpy()   # Confidence scores
                cls_ids = boxes.cls.cpu().numpy().astype(int).tolist()  # Class IDs
                tags_dict = result.names  # Class names dictionary

                # Get the detected class names and database output
                masks_output = [tags_dict[cls_id] for cls_id in cls_ids]
                db_output = list(map(lambda mask: check_(local_list, str(mask)), masks_output))

                # Show the processed frame with bounding boxes
                result_frame = result.plot()
                cv2.imshow("Detection Results", result_frame)
                pt(db_output)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

    def __call__(self) -> None:
        self.process_video()


def train_(model_path: any, data_set_path: any, epochs: int, size: int): #функция тренеровки модели
    model_path.info() # Display model information (optional)
    model_path.train(data=data_set_path, epochs=epochs, imgsz=size) # Train the model on the COCO8 example dataset for 100 epochs

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

def conv_pt_onnx(model): model.export(format='onnx')  # creates onnx model

def conv_pt_engine(model): model.export(format='engine')  # creates onnx model




