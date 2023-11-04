import cv2 as cv
import torch
import torch.nn as nn
import argparse
from torchvision import transforms
import numpy as np
from Model import DeePixBiS
from Loss import PixWiseBCELoss
from Metrics import predict, test_accuracy, test_loss
import torch
from facenet_pytorch import MTCNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
rate = 3
model = DeePixBiS()

name_model = './DeePixBiS.pth'

model.load_state_dict(torch.load(name_model))
model.eval()
tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Return xac suat
def make_predict(img):
    faceRegion = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    faceRegion = tfms(faceRegion)
    faceRegion = faceRegion.unsqueeze(0)

    mask, binary = model.forward(faceRegion)
    res = torch.mean(mask).item()
    return res

def preprocess(img):
    small_frame = cv.resize(
        img, (0, 0), fx=round(1 / rate, 2), fy=round(1 / rate, 2)
    )
    norm_img = np.zeros((small_frame.shape[0], small_frame.shape[1]))
    norm_small_frame = cv.normalize(
        small_frame, norm_img, 0, 255, cv.NORM_MINMAX
    )
    small_rgb_frame = cv.cvtColor(norm_small_frame, cv.COLOR_BGR2RGB)
    return small_rgb_frame

def one_image(img):
    img_copy = img.copy()
    mtcnn = MTCNN()
    small_rgb_frame = preprocess(img)
    try:
        faces, probs = mtcnn.detect(small_rgb_frame)

        for face, prob in zip(faces, probs):
            if prob > 0.5:
                x1, y1, x2, y2 = face
                x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
                X1, Y1, X2, Y2 = (
                    int(x1 * rate),
                    int(y1 * rate),
                    int(x2 * rate),
                    int(y2 * rate),
                )

                face_region = img_copy[Y1:Y2, X1:X2]

                x, y, w, h = X1, Y1, X2 - X1, Y2 - Y1
                # cv.imshow('Test', faceRegion)
                print(x, y, w, h)
                res = make_predict(face_region)
                print("---------")
                print(res)
                if res < 0.5:
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv.putText(img, 'Fake', (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                    print("fake")
                else:
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.putText(img, 'Real', (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
                    print("real")
                print("---------")
                cv.imshow('Deep Pixel-wise Binary Supervision Anti-Spoofing', img)
    except:
        print("Lỗi không tìm thấy mặt")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type_input", default="image", help="Type of input (image/camera)"
    )

    return parser.parse_args()


args = get_parser()


if args.type_input == "image":
    name_img_test = 'drama'
    img = cv.imread(name_img_test + '.jpg')
    one_image(img)
    cv.imwrite("result_" + name_img_test + ".jpg", img)
    cv.waitKey(0)

else:
    while True:
        cap = cv.VideoCapture(0)
        _, img = cap.read()
        one_image(img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
