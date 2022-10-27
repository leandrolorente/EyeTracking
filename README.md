# BlinkDetector

Neste projeto, mostro como construir um rastreador de olhos usando OpenCV, Python e dlib.

O primeiro passo é realizar a detecção da referência facial para localizar os olhos em uma determinada moldura de um vídeo.

Uma vez que temos os marcos faciais para ambos os olhos, calculamos a relação de aspecto do olho para cada olho, o que nos dá um valor único.

Uma vez que temos a proporção de aspecto do olho, podemos definir um limiar para determinar qual direção a pessoa está olhando .
