import dlib
import cv2
import glob
import os

detectorOlhos = dlib.simple_object_detector("recursos/detector_olhos.svm")
detectorPontosOlhos = dlib.shape_predictor("recursos/detector_olhos_pontos.dat")

print(dlib.test_shape_predictor("recursos/teste_olhos_pontos.xml", "recursos/detector_olhos_pontos.dat"))


def imprimirPontos(imagem, pontos):
    for p in pontos.parts():
        cv2.circle(imagem, (p.x, p.y), 2, (0, 255, 0))

detector = dlib.get_frontal_face_detector()

for arquivo in glob.glob(os.path.join("olhos_teste","*.jpg")):
    imagem = cv2.imread(arquivo)
    facesDetectadas = detector(imagem)
    for face  in facesDetectadas:
        e = int(face.left())
        t = int(face.top())
        r = int(face.right())
        b = int(face.bottom())
        crop_img = imagem[t:b,e:r]
        #cv2.imshow("ds",crop_img)
        #cv2.waitKey(0)
        #exit(1)
        objetosDetectados = detectorOlhos(crop_img, 2)
        for olhos in objetosDetectados:
            e, t, d, b = (int(olhos.left()), int(olhos.top()), int(olhos.right()), int(olhos.bottom()))
            cv2.rectangle(crop_img, (e, t), (d, b), (0, 0, 255), 2)
            pontos = detectorPontosOlhos(crop_img, olhos)
            imprimirPontos(crop_img, pontos)


        cv2.imshow("Detector Pontos", crop_img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

