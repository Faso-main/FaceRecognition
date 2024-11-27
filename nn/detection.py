from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
from mtcnn import MTCNN  # Импортируем MTCNN
from db.main import *
from pprint import pprint as pt

class MainField:  # 240ms(fused)
    
    def __init__(self, model, size, source, conf_):
        self.model = self.load_model(model)  # инициализация оптимиз. модели
        self.size = size
        self.source = source  # 0 для дефолт-камеры\ссылка для обработки видео
        self.conf_ = conf_

        self.mtcnn = MTCNN(keep_all=True)  # Инициализация модели MTCNN

    def __call__(self): 
        self.process_video()

    def load_model(self, model):
        model.fuse()  # ускорение работы модели
        return model

    def process_video(self):
        cap = cv2.VideoCapture(self.source)  # доступ к веб-камере (индекс 0)
        while True:  # основной цикл
            ret, frame = cap.read()  # обработка потока веб-камеры
            if not ret: break
            
            # Обнаружение лиц с помощью MTCNN
            faces = self.mtcnn.detect(frame)
            face_boxes = []
            if faces[0] is not None:
                for face in faces[0]:
                    x1, y1, x2, y2 = [int(b) for b in face]
                    face_boxes.append([x1, y1, x2, y2])
                    
            results = self.model.predict(frame, conf=self.conf_, imgsz=self.size)  # параметры модели
            
            dict_ = [result.names for result in results][0]  # словарь структуры dataсета
            xyxys = [result.boxes.cpu().numpy().xyxy for result in results][0]  # xyxy параметры бокса (верхний левый, правый нижний по сетке)
            conf = [result.boxes.cpu().numpy().conf for result in results][0]  # уровни confidence модели
            cls_ids = [result.boxes.cpu().numpy().cls.astype(int).tolist() for result in results][0]  # итерации -> cpu -> nparray -> индексы -> список
            masks_output = [dict_[itr] for itr in cls_ids]  # список лиц по индексу

            db_output = [check_(local_list, str(itr)) for itr in masks_output]  # вывод из базы (google sheets)

            # Рисуем боксы для лиц
            for (x1, y1, x2, y2) in face_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Рисуем бокс

            vd_outp = cv2.imshow("_________", results[0].plot())  # окно вывода
            pt(db_output)  # вывод

            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        self.__del__()

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()


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

def conv_pt_onnx(model): model.export(format='onnx')  # creates onnx model

def conv_pt_engine(model): model.export(format='engine')  # creates onnx model




