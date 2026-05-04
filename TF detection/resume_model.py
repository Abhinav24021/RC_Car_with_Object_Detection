from ultralytics import YOLO
import multiprocessing

def main():
    # Load the partially trained model from the last checkpoint
    model = YOLO(r'runs\detect\train15\weights\last.pt')

    # Resume training with early stopping
    model.train(resume=True, patience=20)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
