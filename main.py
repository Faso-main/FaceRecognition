from nn.detection import MainField,init_model
import os


dateset_path=os.path.join('tasks','view_tasks', 'runs','detect', 'train_50e', 'weights', 'best.pt')

def main():
    print(f'Running Yolo based detection algorithm...........')
    #tasks/view_tasks/runs/detect/train_23d/weights/best.pt
    #train_(init_model('nn/models/yolov8s.pt'),'nn/data_sets/users_23/data.yaml',50,256)
    #conv_pt_onnx(init_model('nn/models/yolov8s.pt'))
    #conv_pt_engine(init_model('nn/models/yolov8s.pt'))
    MainField(init_model(dateset_path),256,0,0.4)()
    #onnx_run2('nn/models/yolov8s.onnx')


try: 
    if __name__ == '__main__': main()
except KeyboardInterrupt: print(f'Stopped by hand......')
except Exception as e:print(f'Error: {e}')
finally: print(f'Done.....')