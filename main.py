#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:38:20 2020

@author: diogo
"""
import face_recognition
import cv2
import numpy as np
import os

menu = {}
menu['1'] = "Adicionar Pessoa"
menu['2'] = "Executar"
menu['0'] = "Sair"

while True:
    options = menu.keys()
    for entry in options:
        print(entry, '->', menu[entry])

    # Menu
    selection = input("Opção:")
    # Adicionar foto
    if selection == '1':
        while True:
            name = input("Nome (0 -> Cancelar):")
            if name == '0':
                break
            elif len(name) < 4:
                print('Nome têm que conter no mínimo três letras')
            else:
                # Video Capture
                cap = cv2.VideoCapture(
                    'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=720, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)

                while(True):
                    ret, frame = cap.read()
                    img_name = "users/{}.png".format(name)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (10, 500)
                    fontScale = 1
                    fontColor = (255, 255, 255)
                    lineType = 2

                    cv2.putText(frame, 'f -> Fotografar',
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                lineType)

                    # Display the captured image
                    cv2.imshow('Adicionar Pessoa', frame)
                    if cv2.waitKey(1) & 0xFF == ord('f'):  # save on pressing 'y'
                        cv2.imwrite(img_name, frame)
                        print('Pessoa', name, 'adicionada com sucesso!')
                        break

                cap.release()
                cv2.destroyAllWindows()

                for i in range(10):
                    cv2.waitKey(1)

                print('')
                break
    elif selection == '2':
        # Reconhecimento
        cap = cv2.VideoCapture(
            'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=720, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)

        # Lista com encoding das fotos guardadas, criada antes da abetura da câmera
        my_dir = 'users/'
        known_face_encodings = []
        known_face_names = []
        for i in os.listdir(my_dir):
            image = my_dir + i
            nome = image.split('/')[1]
            nome = nome.split('.')[0]
            image = face_recognition.load_image_file(
                image)
            image_encoding = face_recognition.face_encodings(
                image)

            if not image_encoding:
                continue

            known_face_encodings.append(image_encoding[0])
            known_face_names.append(nome)

        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True

        while True:
            ret, frame = video_capture.read()

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            rgb_small_frame = small_frame[:, :, ::-1]

            if process_this_frame:
                # Bounding boxes das faces no frame
                face_locations = face_recognition.face_locations(
                    rgb_small_frame)
                # Encoding dos faces encontradas (O método de procura das landmarks é utilizado no método 'face_encodings')
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, face_locations)

                face_names = []
                # Comparação de cada face_encoding com encoding das fotos
                for face_encoding in face_encodings:
                    # UTILIZADO APENAS PARA DEMONSTRAÇÃO VISUAL
                    """ Comparação entre distância entre encoding do frame atual e encodings das fotos guardadas,
                        Retorna uma lista com o número de elemtnos igual ao número de 'known_face_encodings' com
                        'True' nas posições em que a distância entre o encoding e known_face_encoding dessa 
                        posição seja inferior a .6 e 'False' caso seja acima.
                    """
                    matches = face_recognition.compare_faces(
                        known_face_encodings, face_encoding)
                    name = "Unknown"

                    # Cálculo de distâcia euclidiana entre encoding e known_face_encodings e verifica se o índice do valor mais baixo corresponde à posição onde a comparação foi True
                    face_distances = face_recognition.face_distance(
                        known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    # Verificação para a abetura de porta - Se alguma distância é inferior a 0.35
                    if matches[best_match_index]:
                        if(face_distances[best_match_index] < 0.35):
                            print('PODE ABRIR A FECHADURA, DISTÂNCIA:',
                                  face_distances[best_match_index])
                        else:
                            print('RECONHECIDO, MAS NÃO O SUFICIENTE PARA ABRIR A FECHADURA, DISTÂNCIA:',
                                  face_distances[best_match_index])

                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Desenho dos retângulos e adição de nome
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top),
                              (right, bottom), (0, 0, 255), 2)

                cv2.rectangle(frame, (left, bottom - 35),
                              (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6),
                            font, 1.0, (255, 255, 255), 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 500)
            fontScale = 1
            fontColor = (255, 255, 255)
            lineType = 2

            cv2.putText(frame, '0 -> Sair',
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

            cv2.imshow('Reconhecimento', frame)

            # 'q' para desligar
            if cv2.waitKey(1) & 0xFF == ord('0'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

        for i in range(10):
            cv2.waitKey(1)

        print('')
        break

    elif selection == '0':
        break
    else:
        print("Opção Inválida!")
