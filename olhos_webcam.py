import sys
import dlib
import cv2
import numpy as np
import imutils
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils

##pula_quadros = 30
##captura = cv2.VideoCapture(0)
##captura.set(cv2.CAP_PROP_FRAME_WIDTH,200)
##captura.set(cv2.CAP_PROP_FRAME_HEIGHT,200)
##contadorQuadros = 0
detector = dlib.simple_object_detector("recursos/detector_olhos.svm")
detectorFace = dlib.get_frontal_face_detector()
detectorPontosOlhos = dlib.shape_predictor("recursos/detector_olhos_pontos.dat")
vs = VideoStream(src=0).start()
fileStream = False

def imprimirPontos(imagem, pontos):
    for p in pontos.parts():
        cv2.circle(imagem, (p.x, p.y), 2, (0, 255, 0))
while True:
    ##conectado, frame = captura.read()

    frame = vs.read()
    frame = imutils.resize(frame, 450, 550)
   ## frame = imutils.resize(frame, width=300, height= 300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    facesDetectadas = detectorFace(gray)
    for face in facesDetectadas:
        e = int(face.left())
        t = int(face.top())
        r = int(face.right())
        b = int(face.bottom())
       ## cv2.rectangle(frame, (e, t), (r, b), (0, 255, 255), 2)
        crop_img = frame[t:b, e:r]
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        objetosDetectados = detector(gray)

        for o in objetosDetectados:
            e, t, d, b = (int(o.left()),int(o.top()),int(o.right()),int(o.bottom()))
            cv2.rectangle(crop_img,(e, t), (d, b), (0, 0, 255), 2)
            pontos = detectorPontosOlhos(crop_img,o)
            imprimirPontos(crop_img,pontos)
            cv2.imshow("Preditor de Olhos", frame)
        if cv2.waitKey(1) & 0xFF == 27:
               break

##captura.release()
vs.stop()
cv2.destroyAllWindows()

sys.exit(0)