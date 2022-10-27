import dlib
import cv2
import os
import glob
import numpy as np

arquivo = open("direcao_olhar.txt","r")
font = cv2.FONT_HERSHEY_SIMPLEX
lista = arquivo.readlines()
alt,larg = lista[0].split(";")
canvas = np.ones((int(alt),int(larg),3))*255
listax = list()
listay = list()

for i in range(1,len(lista)):
    #print(lista[i])
    x, y, texto = lista[i].split(";")
    #if texto == "CENTRO\n":
    listax.append(int(x))
    listay.append(int(y))

soma_x = np.sum(listax)
qtde_x = len(listax)
soma_y = np.sum(listay)
qtde_y = len(listay)

media_x = soma_x/qtde_x
media_y = soma_y/qtde_y


cv2.circle(canvas,(int(media_x ),int(media_y)), 1, (0, 0, 0), 1)

for i in range(1,len(lista)):
    #print(lista[i])
    x, y, texto = lista[i].split(";")
    cv2.circle(canvas,(int(x),int(y)),10,(0,0,0),1)

    cv2.putText(canvas,texto,(50, 100), font, 1, (0, 0, 255), 1)
    cv2.imshow("Canvas", canvas)
    key = cv2.waitKey(100000)
    if key == 13:
        canvas = np.ones((int(alt),int(larg),3))*255
        cv2.circle(canvas, (int(media_x),int(media_y)), 1, (0, 0, 0), 1)

cv2.release()
cv2.destroyAllWindows()
arquivo.close()

