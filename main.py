import numpy as np
import model as Model
import torch
import utils
from facenet_pytorch import MTCNN
from facenet_pytorch.models.utils.detect_face import crop_resize, get_size
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import glob
import re
import ffmpeg
from argparse import ArgumentParser

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 390)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2
img_path = 'images/'


def extract_features(weights_path):
    with torch.no_grad():
        model = Model.VGGFace_Extractor().to(device)
        model.load_state_dict(torch.load(weights_path))
        model.eval()
        for filename in os.listdir(img_path):
            # print(filename)
            img = Image.open(img_path + filename).convert('RGB')
            features = model(utils.preprocess(
                img, device).reshape(-1, 3, 224, 224))
            torch.save(features, 'extracted_features/' +
                       filename.split('.')[0] + '.pt')
            print(f'Features from {filename} extraced!')


def check_video(weights_path, margins=40, facenet_threshold=.985, euclidean_distance_threshold=120.0):
    with torch.no_grad():
        mtcnn = MTCNN(image_size=256, margin=0)
        model = Model.VGGFace_Extractor().to(device)
        model.load_state_dict(torch.load(weights_path))
        model.eval()

        cap = cv2.VideoCapture(0)

        while (cap.isOpened()):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.imshow('frame', frame)
            if not (ret):
                break

            boxes, probs = mtcnn.detect(frame)
            print(boxes, probs)
            img_draw = frame.copy()
            img_draw = Image.fromarray(img_draw)
            # draw = ImageDraw.Draw(img_draw)

            if boxes is not None:
                names = []
                distances_difference = []
                for (box, point) in zip(boxes, probs):
                    if point < facenet_threshold:
                        continue
                    margin = margins
                    image_size = 256
                    margin = [
                        margin * (box[2] - box[0]) / (image_size - margin),
                        margin * (box[3] - box[1]) / (image_size - margin)
                    ]
                    raw_image_size = get_size(frame)
                    box = [
                        int(max(box[0] - margin[0] / 2, 0)),
                        int(max(box[1] - margin[1] / 2, 0)),
                        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
                        int(min(box[3] + margin[1] / 2, raw_image_size[1]))
                    ]

                    face = img_draw.crop(box).copy().resize(
                        (image_size, image_size), Image.BILINEAR).convert('RGB')
                    # print(type(face))
                    features_1 = model(utils.preprocess(
                        face, device).reshape(-1, 3, 224, 224))
                    images_path = 'extracted_features/'
                    data_path = os.path.join(images_path, '*pt')
                    files = glob.glob(data_path)
                    name = 'Unkown'
                    best_distance = euclidean_distance_threshold + 5
                    for i, f1 in enumerate(files):
                        print('Entro aqui')
                        features = torch.load(f1)

                        distance = utils.euclidean_distance(
                            features, features_1)
                        print(
                            f'File - {f1.split(".")[0]} - Distance {distance}')
                        if distance < euclidean_distance_threshold and distance < best_distance:
                            best_distance = distance
                            name = (f1.split('.')[0]).split('\\')[-1]
                    names.append(name)
                    distances_difference.append(best_distance)

                for (box, point, name, distances) in zip(boxes, probs, names, distances_difference):
                    if point < facenet_threshold or name == "Unknown":
                        continue
                    # print(box)
                    # print((box[0], box[1]), (box[2], box[3]))
                    cv2.rectangle(
                        frame, (int(box[0]-25), int(box[1]-25)), (int(box[2]+25), int(box[3]+25)), (255, 255, 255), 5)
                    cv2.putText(frame, name, (box[0], int(box[1]-40)), font,
                                fontScale,
                                fontColor,
                                lineType)

            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            if key == ord('0'):
                break
        cap.release()
        cv2.destroyAllWindows()


def openCam(add=False):
    if add:
        name = input('Nome:\n')

    video_capture = cv2.VideoCapture(0)
    # print(f'FPS {video_capture.get(cv2.CAP_PROP_FPS)} | WIDTH {video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)} | HEIGHT {video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)} | BACKEND {video_capture.get(cv2.CAP_PROP_BACKEND)}')

    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        # frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)
        cv2.putText(frame, '0 -> Sair',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        cv2.putText(frame, 'f -> Fotografar',
                    (10, 360),
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        cv2.rectangle(frame, (int(frame.shape[1] * 0.30), int(frame.shape[0]*0.05)),
                      (int(frame.shape[1] * 0.70), int(frame.shape[0]*0.70)), (0, 255, 0))

        cv2.imshow('Frame', frame)

        key = cv2.waitKey(1)
        if key == ord('f'):
            # print('Entrei no f')
            nums = []
            for filename in os.listdir(img_path):
                if name in filename and filename.endswith('.png'):
                    nums.append(int((filename.split('_')[-1]).split('.')[0]))

            if len(nums) > 0:
                name_path = img_path + name + f'_{int(max(nums))+1}.png'
            else:
                print('Entro ')
                name_path = img_path + name + '_1.png'

            print(f'Imagem gravada com nome {name_path}')
            cv2.imwrite(name_path, frame[int(frame.shape[0]*0.05)+1:int(
                frame.shape[0]*0.70), int(frame.shape[1] * 0.30)+1:int(frame.shape[1] * 0.70)])

        elif key == ord('0'):
            # print('Sair')
            break

    video_capture.release()
    cv2.destroyAllWindows()


def main():
    while True:
        print('1 - Adicionar foto')
        print('2 - Cam')
        print('3 - Vid')
        print('0 - Sair')
        opt = input('Opção:\n')

        if opt == '1':
            openCam(add=True)
        elif opt == '2':
            extract_features('models/face_extractor_model.mdl')
        elif opt == '3':
            check_video('models/face_extractor_model.mdl')
        elif opt == '0':
            exit()
        else:
            print('Opção inválida ')


if __name__ == '__main__':
    main()
