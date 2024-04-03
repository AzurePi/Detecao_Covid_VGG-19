# -*- coding: utf-8 -*-

# Importações ----------------------------------------------------------------------------------------------------------

import os
import shutil

from keras.callbacks import CSVLogger
import tensorflow as tf

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

import random
import datetime

"""# Preparação do Dataset"""

# @title Download a partir do Kaggle e formatação do diretóro do dataset  ----------------------------------------------
'''
  Se o dataset ainda não foi baixado, baixamos e descompactamos
  Mantemos apenas as imagens originais, e não as pré-processadas (faremos o pré-processamento nós mesmos)
'''
if not os.path.exists("./dataset"):
    os.mkdir("./dataset")

    os.system('kaggle datasets download -d azaemon/preprocessed-ct-scans-for-covid19 -p ./dataset --unzip -q')
    shutil.rmtree("./dataset/Preprocessed CT scans")

    directories = [d for d in os.listdir("./dataset/Original CT Scans") if
                   os.path.isdir(os.path.join("./dataset/Original CT Scans", d))]

    for directory in directories:
        src = os.path.join("./dataset/Original CT Scans", directory)
        dest = os.path.dirname("./dataset/Original CT Scans")
        print(dest)
        shutil.move(src, dest)
    os.rmdir("./dataset/Original CT Scans")

    for file_name in os.listdir("./dataset/NiCT"):
        src = os.path.join("./dataset/NiCT", file_name)
        dest = os.path.join("./dataset/nCT", file_name)
        shutil.move(src, dest)
    os.rmdir("./dataset/NiCT")

    os.rename("./dataset/nCT", "./dataset/negative")
    os.rename("./dataset/pCT", "./dataset/positive")

# @title Restrição do tamanho do dataset -------------------------------------------------------------------------------
# utilizando um sampling randomizado, separamos as imagens em conjuntos de treinamento e validação

sampling_training = 0.3
sampling_validation = 0.3

os.mkdir("./temp")
shutil.copytree("./dataset", "./temp", dirs_exist_ok=True)

if not (os.path.exists("./training")):
    os.mkdir("./training")
    os.mkdir("./training/positive")
    os.mkdir("./training/negative")

    # selecionamos uma fração das imagens positivas para treinamento
    diretorio = os.listdir("./temp/positive")
    diretorio = random.sample(diretorio, int(len(diretorio) * sampling_training))
    source = "./temp/positive/"
    for imagem in diretorio:
        shutil.move(f'{source}{imagem}', "./training/positive/")

    # selecionamos uma fração das imagens negativas para treinamento
    diretorio = os.listdir("./temp/negative")
    diretorio = random.sample(diretorio, int(len(diretorio) * sampling_training))
    source = "./temp/negative/"
    for imagem in diretorio:
        shutil.move(f'{source}{imagem}', "./training/negative/")

if not (os.path.exists("./validation")):
    os.mkdir("./validation")
    os.mkdir("./validation/positive")
    os.mkdir("./validation/negative")

    # selecionamos uma fração das imagens positivas para validação
    diretorio = os.listdir("./temp/positive")
    diretorio = random.sample(diretorio, int(len(diretorio) * sampling_validation))
    source = "./temp/positive/"
    for imagem in diretorio:
        shutil.move(f'{source}{imagem}', "./validation/positive/")

    # selecionamos uma fração das imagens negativas para validação
    diretorio = os.listdir("./temp/negative")
    diretorio = random.sample(diretorio, int(len(diretorio) * sampling_validation))
    source = "./temp/negative/"
    for imagem in diretorio:
        shutil.move(f'{source}{imagem}', "./validation/negative/")

shutil.rmtree("./temp")

# @title Criação de dataset keras --------------------------------------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory="./training",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(256, 256))

validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory="./validation",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(256, 256))

"""# Aplicação da VGG-19"""

image_input = tf.keras.layers.Input(shape=(256, 256, 3))
VGG_19_base = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_tensor=image_input,
                                          pooling="max", classifier_activation="softmax")

for i, layer in enumerate(VGG_19_base.layers):
    layer.trainable = False

print(VGG_19_base.summary())

# @title Layers para fine-tuning ---------------------------------------------------------------------------------------
FC_layer_Flatten = tf.keras.layers.Flatten()(VGG_19_base.output)
Dense = tf.keras.layers.Dense(units=256, activation="relu")(FC_layer_Flatten)
Dense = tf.keras.layers.Dense(units=128, activation="relu")(Dense)
Dense = tf.keras.layers.Dense(units=64, activation="relu")(Dense)
Dense = tf.keras.layers.Dense(units=32, activation="relu")(Dense)
Dense = tf.keras.layers.Dense(units=16, activation="relu")(Dense)
Classification = tf.keras.layers.Dense(units=2, activation="softmax")(Dense)

modelo_final = tf.keras.Model(inputs=image_input, outputs=Classification)
modelo_final.summary()

base_learning_rate = 0.001
epocas = 100

modelo_final.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                     loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

csv_logger = CSVLogger("training.csv", ",", False)  # cria um log em CSV para armazenar a história do modelo

start_time = datetime.datetime.now()
history = modelo_final.fit(x=train_ds, epochs=epocas, validation_data=validation_ds, callbacks=[csv_logger], verbose=2)
end_time = datetime.datetime.now()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Acurácia do Modelo")
plt.xlabel("Épocas")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # garante que só inteiros serão utilizados na escala
plt.ylabel("Acurácia")
plt.legend(['train', 'validation'])
plt.savefig("acuracia.png", bbox_inches='tight')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Perda do Modelo")
plt.xlabel("Época")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # garante que só inteiros serão utilizados na escala
plt.ylabel("Perda")
plt.legend(['train', 'validation'])
plt.savefig("perda.png", bbox_inches='tight')
plt.close()

with open("parametros.txt", "w") as file:
    file.write(f"Sampling do training set: {sampling_training}")
    file.write(f"Sampling do training set: {sampling_validation}")
    file.write(f"Learning rate: {base_learning_rate}")
    file.write(f"Épocas: {epocas}")
    file.write(f"Tempo decorrido: {end_time-start_time}")
