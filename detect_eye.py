import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import string
from math import hypot

arquivo_direcoes = open("direcao_olhar.txt","w")
direcoes = list()
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")

def pontoMedio(p1,p2):
	return int((p1.x + p2.x)/2), int((p1.y+p2.y)/2)

font = cv2.FONT_HERSHEY_SIMPLEX

def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]

def get_blinking_ratio(eye_points, facil_landmarks):
	left_point = (facil_landmarks.part(eye_points[0]).x, facil_landmarks.part(eye_points[0]).y)
	right_point = (facil_landmarks.part(eye_points[3]).x, facil_landmarks.part(eye_points[3]).y)
	center_top = pontoMedio(facil_landmarks.part(eye_points[1]), facil_landmarks.part(eye_points[2]))
	center_bottom = pontoMedio(facil_landmarks.part(eye_points[5]), facil_landmarks.part(eye_points[4]))
	##hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
	##ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

	hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
	ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

	ratio = hor_line_lenght / ver_line_lenght
	return ratio

def get_gaze_ratio(eye_points,facial_landmarks,thresh,detectorBlob,blinking_ratio):
	right_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
								 (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
								 (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
								 (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
								 (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
								 (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], dtype=np.int32)

	# cv2.polylines(frame, [right_eye_region], 1, (0, 0, 255), 2,)
	##print(left_eye_region)
	height, width, _ = frame.shape
	mask = np.zeros(frame.shape[:2], np.uint8)

	mask2 = np.zeros(frame.shape[:2], np.uint8)
	#for i in range (0,len(mask)):
	#	for j in range(0,len(mask[i])):
	#		mask[i][j] = 255;

	#cv2.imshow("mascara", mask)
	cv2.polylines(mask, [right_eye_region], True, 255, 2)
	cv2.fillPoly(mask, [right_eye_region], 255)
	mask = cv2.dilate(mask, kernel, 13)



	cv2.polylines(mask2, [right_eye_region], True, 255, 2)
	cv2.fillPoly(mask2, [right_eye_region], 255)
	kernel2 = np.ones((21, 21), np.uint8)
	mask2 = cv2.dilate(mask2, kernel2, 15)
	
	##mask = cv2.erode(mask, kernel, 4)
	#cv2.imshow("mascara dilatada", mask)

	eye = cv2.bitwise_and(frame, frame, mask=mask)
	#cv2.imshow("eye_region", eye)
	#cv2.waitKey(10000000)
	eye2 = cv2.bitwise_and(frame, frame, mask=mask2)
	#cv2.imshow("eye", eye)


	##key = cv2.waitKey(10000)
	##if key == 27:
	##	pass;

	min_x = np.min(right_eye_region[:, 0])
	max_x = np.max(right_eye_region[:, 0])
	min_y = np.min(right_eye_region[:, 1])
	max_y = np.max(right_eye_region[:, 1])

	gray_eye = eye[min_y: max_y, min_x: max_x]

	##eye = frame[min_y: max_y, min_x: max_x]
	##cv2.imshow("eye2", eye)

	# = cv2.cvtColor(olho_recortado,cv2.COLOR_BGR2GRAY)
	##cv2.imshow("gray eye recortado primerio ", gray_eye)
	##key = cv2.waitKey(10000)
	##if key == 27:
	##	pass;
	##gray_eye = cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
	##cv2.imshow("gray_eye", eye)
	##gray_eye = cv2.GaussianBlur(eye,(7,7),255)
	##cv2.imshow("gray_eye_gausian", gray_eye)
	#gray_eye = cv2.equalizeHist(gray_eye)
	#gray_eye = cv2.dilate(gray_eye, None, iterations=1)
	gray_eye = cv2.cvtColor(gray_eye,cv2.COLOR_BGR2GRAY)
	#cv2.imshow("eye framezado ", gray_eye)
	eye = cv2.cvtColor(eye2,cv2.COLOR_BGR2GRAY)
	#cv2.imshow("gray eye pos cinza ", gray_eye)
	##key = cv2.waitKey(10000)
	##if key == 27:
	##	pass;

	threshold = cv2.getTrackbarPos('threshold', 'FRAME')
	_, threshold_eye = cv2.threshold(eye, threshold , 255, cv2.THRESH_BINARY)
	_, thresh = cv2.threshold(gray_eye, threshold, 255, cv2.THRESH_BINARY)
	#cv2.imshow("gray eye recortado pos threshold ", thresh)
	##key = cv2.waitKey(10000)
	##if key == 27:
	##	pass;

	##threshold_eye = cv2.medianBlur(threshold_eye,3)
	##_,threshold_eye2 = cv2.threshold(threshold_eye, 58, 255, cv2.THRESH_BINARY)
	##thresh = cv2.erode(threshold_eye, None, iterations=1)
	##thresh = cv2.dilate(threshold_eye, None, iterations=1)
	#threshold_eye = cv2.erode(threshold_eye, None, iterations=2)  # 1
	#threshold_eye = cv2.dilate(threshold_eye, None, iterations=4)  # 2
	#cv2.imshow("antes dilatar ", thresh)
	#key = cv2.waitKey(10000)
	#if key == 27:
	#	pass;


	#threshold_eye = cv2.dilate(threshold_eye, kernel, iterations=1)
	#cv2.imshow("depois dilatar ", thresh)
	#key = cv2.waitKey(10000)
	#if key == 27:
	#	pass;

	#thresh = cv2.erode(thresh, kernel, iterations=1)
	#thresh = cv2.dilate(thresh, kernel, iterations=1)
	thresh = cv2.medianBlur(thresh,7)  #3


	#threshold_eye = cv2.erode(threshold_eye, kernel2, iterations=1)
	#threshold_eye = cv2.dilate(threshold_eye, kernel2, iterations=1)
	threshold_eye = cv2.medianBlur(threshold_eye,7)
	#threshold_eye = cv2.dilate(threshold_eye, kernel2, iterations=1)
	#_, threshold_eye = cv2.threshold(eye, threshold/2, 255, cv2.THRESH_BINARY)
	#threshold_eye = cv2.medianBlur(threshold_eye, 5)
	##threshold_eye = cv2.dilate(threshold_eye, None, iterations=2)
	#threshold_eye = cv2.erode(threshold_eye, kernel2, iterations=1)
	##threshold_eye = cv2.bitwise_not(threshold_eye)
	##thresh = cv2.bitwise_not(thresh)
	#contornos = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	# for cnt in contornos:
	#	cv2.drawContours(frame,[cnt], -1, (0,0,255),1)
	##print (contornos)
	##threshold_eye = cut_eyebrows(threshold_eye)
	keypoints = detectorBlob.detect(threshold_eye)
	listaPontos = list()
	str = ""
	x , y = 0,0
	for k in keypoints:
		x,y = np.int(k.pt[0]),np.int(k.pt[1])
		direct = repr(x) +";"+ repr(y)
		if x != 0 and x is not None and y != 0 and y is not None:
			listaPontos.append(direct)
		cv2.circle(frame,(x,y),1,255,-1)
		#print(x,y)
	#print(keypoints.pt)
	#cv2.putText(frame,".",(int(x+5),int(y+3)),font,1,(0,0,255),1)

	cv2.drawKeypoints(frame, keypoints, frame, (0, 0, 255),print(cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
	 #print(cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	#print(t)
	#cv2.keypo
	##for cnt in contornos:
	##	cv2.drawContours(eye,[cnt], -1, (0,0,255), 3)

	##thresh = cv2.bitwise_not(thresh)
	##thresh = cv2.medianBlur(thresh, 3)

	##key = cv2.waitKey(10000)
	##if key == 27:
	##	pass;
	height, width =  thresh.shape

	##lado esquerdo do olho
	left_side_threshold =  thresh[0: height, 0: int(width / 2)]
	left_side_white = cv2.countNonZero(left_side_threshold)

	##lado esquerdo cima do olho
	left_up_side_threshold = thresh[0: int(height/2), 0: int(width / 2)]
	left_up_side_white = cv2.countNonZero(left_up_side_threshold)

	##lado esquerdo baixo do olho
	left_down_side_threshold = thresh[int(height/2): height, 0: int(width / 2)]
	left_down_side_white = cv2.countNonZero(left_down_side_threshold)

	##lado direito do olho
	right_side_threshold = thresh[0: height, int(width / 2):width]
	right_side_white = cv2.countNonZero(right_side_threshold)

	##lado direito cima do olho
	right_up_side_threshold = thresh[0: int(height/2), int(width / 2):width]
	right_up_side_white = cv2.countNonZero(right_up_side_threshold)

	##lado direito baixo do olho
	right_down_side_threshold = thresh[int(height/2): height, int(width / 2):width]
	right_down_side_white = cv2.countNonZero(right_down_side_threshold)

	#cv2.circle(threshold_eye, (cx, cy), 4, (0, 0, 255), 2)



	#cv2.imshow("EYE",eye)
	####	cv2.drawContours(eye,[cnt],-1,(0,0,255),3)
	#cv2.imshow("gray eye recortado finaçl ", thresh)
	#cv2.imshow("Threshold + dilatacao",thresh)
	#cv2.imshow("Threshold", threshold_eye)
	##cv2.imshow("FRAMEZERA", frame)
	#cv2.imshow("Threshold2", threshold_eye2)
	#cv2.imshow("mask", gray_eye)

	cima = False
	baixo= False
	esquerdo= False
	direito= False
	esq= False
	dir= False
	up= False
	down= False

	if right_side_white == 0:
			aux = left_side_white/0.0000001
	else:
			aux = left_side_white / right_side_white

	if right_up_side_white == 0:
		aux2 = left_up_side_white / 0.0000001
	else:
		aux2 = left_up_side_white / right_up_side_white



	if 0.9 <= aux <= 1.35:
		if 0.9<= aux <= 1.1 and  (right_down_side_white == right_down_side_threshold.size) and (left_down_side_white == left_down_side_threshold.size) :
			up = True
		elif 0.9 <= (aux2) <=1.35 and blinking_ratio >=3 :
			down = True
		else:
			center = True
	else:
		if left_side_white < right_side_white:
			esquerdo = True
			if left_down_side_white == 0:
				aux = left_up_side_white / 0.0000001
			else:
				aux = left_up_side_white / left_down_side_white
			if 0.6 <= aux <= 1.3:
				esq = True
			else:
				if left_up_side_white < left_down_side_white and blinking_ratio <=3.5:
					menorLado = left_up_side_white
					maiorLado = left_down_side_white
					cima = True
				else:
					menorLado = left_down_side_white
					maiorLado = left_up_side_white
					baixo = True
		else:
			if right_down_side_white == 0:
				aux = right_up_side_white / 0.0000001
			else:
				aux = right_up_side_white / right_down_side_white
			if 0.6 <= aux <= 1.3:
				dir = True
			else:
				direito = True
				if right_up_side_white < right_down_side_white and blinking_ratio < 2.5:
					menorLado = right_up_side_white
					maiorLado = right_down_side_white
					cima = True
				else:
					menorLado = right_down_side_white
					maiorLado = right_up_side_white
					baixo = True

	gaze_ratio = 9
	if (not esq and not dir and esquerdo):
			if cima:
				gaze_ratio = 1 ##diagonal esquerda cima
			else:
				gaze_ratio = 2  ##diagonal esquerda baixo
	elif esq:
		gaze_ratio = 3 ## ESQUERDA
	elif (not dir and direito):
			if cima:
				gaze_ratio = 4 ##diagonal direita cima
			else:
				gaze_ratio = 5 ##diagonal direita baixo
	elif dir:
		gaze_ratio = 6
	elif not dir and not esq:
		if 0.7 <= aux <= 1.35:
			if up:
				gaze_ratio = 7 ## cima
			elif down:
				gaze_ratio = 8 ## baixo
			else:
				gaze_ratio = 9 ## center

##	if right_side_white == 0:
##		gaze_ratio = left_side_white / 1
##	else:
##		gaze_ratio = left_side_white / right_side_white

	key = cv2.waitKey(1)
	tupla = (gaze_ratio,thresh,keypoints),aux,listaPontos,key
	return tupla

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
    except:
        pass
def nothing(x):
    pass

kernel = np.ones((9, 9), np.uint8)
cv2.namedWindow('FRAME')
cv2.createTrackbar('threshold', 'FRAME', 0, 255,nothing)
cv2.setTrackbarPos('threshold', 'FRAME',21)
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea  = 1500
detector_params.minArea  = 50
detector_params.filterByConvexity = True;
detector_params.minConvexity = 0.5
direcoes = list()

detectorBlob = cv2.SimpleBlobDetector_create(detector_params)
control = 0

while True:
	_,frame = cap.read()
	if control == 0:
		alt,larg,_ = frame.shape
		arquivo_direcoes.write(repr(alt)+";"+repr(larg))
		control = 1

	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	thresh = frame.copy()
	faces = detector(gray)
	for face in faces:
		#x,y = face.left(),face.top()
		#x1,y1 = face.right(),face.bottom()
		##cv2.rectangle(frame,(x,y),(x1,y1),(0,255,0),2)

		landmarks = predictor(gray,face)


		#deteccao de piscadas
		left_eye_ratio = get_blinking_ratio([36,37,38,39,40,41],landmarks)
		right_eye_ratio = get_blinking_ratio([42,43,44,45,46,47],landmarks)
		blinking_ratio = (left_eye_ratio + right_eye_ratio)/2 ## aqui permite que se conte a piscada so se for os dois olhos

		#if blinking_ratio > 5.7:
		#	cv2.putText(frame,"BLINKING",(50,150),font,3,(255,0,0))


		## deteccao de olhar
		gaze_ratio_left_eye,numero, lista,key = get_gaze_ratio([36,37,38,39,40,41],landmarks,thresh,detectorBlob,blinking_ratio)
		gaze_ratio_right_eye, _, _,_= get_gaze_ratio([42,43,44,45,46,47],landmarks,thresh,detectorBlob,blinking_ratio)
		gaze = gaze_ratio_left_eye;
		aux = list()






		gaze_ratio = gaze[0]
		thresh = gaze[1]
		texto_direcao =""
		if gaze_ratio == 1:
			cv2.putText(frame,"DIAGONAL DIREITA CIMA",(50,100),font,1,(0,0,255),3)
			texto_direcao = "DIAGONAL DIREITA CIMA"
		elif gaze_ratio == 2:
			cv2.putText(frame,"DIAGONAL DIREITA BAIXO",(50,100),font,1,(0,0,255),3)
			texto_direcao = "DIAGONAL DIREITA BAIXO"
		elif gaze_ratio == 3:
			cv2.putText(frame,"DIREITA",(50,100),font,2,(0,0,255),3)
			texto_direcao = "DIREITA"
		elif gaze_ratio == 4:
			cv2.putText(frame,"DIAGONAL ESQUERDA CIMA",(50,100),font,1,(0,0,255),3)
			texto_direcao = "DIAGONAL ESQUERDA CIMA"
		elif gaze_ratio == 5:
			cv2.putText(frame,"DIAGONAL ESQUERDA BAIXO",(50,100),font,1,(0,0,255),3)
			texto_direcao = "DIAGONAL ESQUERDA BAIXO"
		elif gaze_ratio == 6:
			cv2.putText(frame,"ESQUERDA",(50,100),font,2,(0,0,255),3)
			texto_direcao = "ESQUERDA"
		elif gaze_ratio == 7:
			cv2.putText(frame, "CIMA", (50, 100), font, 2, (0, 0, 255), 3)
			texto_direcao = "CIMA"
		elif gaze_ratio == 8:
			cv2.putText(frame, "BAIXO", (50, 100), font, 2, (0, 0, 255), 3)
			texto_direcao = "BAIXO"
		elif gaze_ratio == 9:
			cv2.putText(frame, "CENTRO", (50, 100), font, 2, (0, 0, 255), 3)
			texto_direcao = "CENTRO"

		for i in range(0,len(lista)):
			direcao = lista[i] + ";"+texto_direcao + "\n"
			direcoes.append(direcao)
		#cv2.putText(frame, str(numero), (50, 150), font, 3, (255, 0, 0))

		#cv2.putText(frame,str(left_side_white),(50,100),font,2,(0,0,255),3)
		#cv2.putText(frame, str(gaze_ratio), (50, 150), font, 2, (0, 0, 255), 3)

		##eye = cv2.resize(gray_eye, None, fx=5, fy=5)

		##cv2.imshow("EYE",eye)
		#cv2.imshow("Threshold",threshold_eye)
		#cv2.imshow("mask", mask)
		#cv2.imshow("eye", frame)

	##print(hor_line_lenght/ver_line_lenght)


	#x = landmarks.part(36).x
	#y = landmarks.part(36).y
	#cv2.circle(frame,(x,y),3,(0,0,255),2)
	#print(landmarks.parts(36))## caso queira pegar a posicao de um ponto especifico apenas usar esse comando parts que retorna a posicao nos eixos
	#print(face)

		##cv2.circle(frame, (mid-8,mid+8), 3, (0, 0, 255), 2)
		shape = shape_to_np(landmarks)
		mid = (shape[42][0] + shape[39][0])
		##thresh = cv2.bitwise_not(thresh)
		#thresh = gaze_ratio_left_eye[1]
		#contouring(thresh[:, 0:mid], mid, frame)
		#thresh = gaze_ratio_right_eye[1]
		#contouring(thresh[:, mid:], mid, frame, True)
		##cv2.drawKeypoints(frame, gaze_ratio_left_eye[2], frame, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		##cv2.drawKeypoints(frame, gaze_ratio_right_eye[2], frame, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

		##contorno = cv2.findContours(gaze_ratio_left_eye[1],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		##contorno2 = cv2.findContours(gaze_ratio_right_eye[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		##frame = cv2.drawContours(frame,contorno,0,(0,255,255),3)
		##frame = cv2.drawContours(frame,contorno2,0,(0,255,255),3)

	cv2.imshow("FRAME",frame)
	#cv2.imshow("image", thresh
	key = cv2.waitKey(1)
	if key ==27:
		break;



for i in range(0,len(direcoes)):
	arquivo_direcoes.write(direcoes[i])
cap.release()
cv2.destroyAllWindows()
arquivo_direcoes.close()
