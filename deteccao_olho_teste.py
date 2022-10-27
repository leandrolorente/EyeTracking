import os
import dlib
import cv2
import glob  ## ler imgs do diretorio

print(dlib.test_simple_object_detector("recursos/olhos_completo.xml","recursos/detector_olhos_completo.svm"))

detectorOlhos = dlib.simple_object_detector("recursos/detector_olhos_completo.svm")

for imagem in glob.glob(os.path.join("olhos_teste","*.jpg")):
    img = cv2.imread(imagem)
    objetosDetectados = detectorOlhos(img)

    for d in objetosDetectados:
        e,t,d,b = (int(d.left()),int(d.top()),int(d.right()),int(d.bottom()))
        cv2.rectangle(img, (e,t), (d, b), (0, 0,255), 2)

    cv2.imshow("Detector Olhos", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()


