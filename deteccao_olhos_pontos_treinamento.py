import dlib

opcoes = dlib.shape_predictor_training_options()
dlib.train_shape_predictor("recursos/olhos_treinamento_novo.xml","recursos/detector_olhos_pontos_novo.dat", opcoes)

