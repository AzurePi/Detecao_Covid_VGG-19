# -*- coding: utf-8 -*-

# Importações ----------------------------------------------------------------------------------------------------------

import os
import shutil
import tensorflow as tf
from matplotlib import pyplot as plt
import random

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
sampling_validation = 0.2

if not (os.path.exists("./training")):
    os.mkdir("./training")
    os.mkdir("./training/positive")
    os.mkdir("./training/negative")

    # selecionamos uma fração das imagens positivas para treinamento
    dir = os.listdir("./dataset/positive")
    dir = random.sample(dir, (int)(len(dir) * sampling_training))
    source = "./dataset/positive/"
    for imagem in dir:
        shutil.copy(source + imagem, "./training/positive/")

    # selecionamos uma fração das imagens negativas para treinamento
    dir = os.listdir("./dataset/negative")
    dir = random.sample(dir, (int)(len(dir) * sampling_training))
    source = "./dataset/negative/"
    for imagem in dir:
        shutil.copy(source + imagem, "./training/negative/")

if not (os.path.exists("./validation")):
    os.mkdir("./validation")
    os.mkdir("./validation/positive")
    os.mkdir("./validation/negative")

    # selecionamos uma fração das imagens positivas para validação
    dir = os.listdir("./dataset/positive")
    dir = random.sample(dir, (int)(len(dir) * sampling_validation))
    source = "./dataset/positive/"
    for imagem in dir:
        shutil.copy(source + imagem, "./validation/positive/")

    # selecionamos uma fração das imagens negativas para validação
    dir = os.listdir("./dataset/negative")
    dir = random.sample(dir, (int)(len(dir) * sampling_validation))
    source = "./dataset/negative/"
    for imagem in dir:
        shutil.copy(source + imagem, "./validation/negative/")


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
                                          pooling="max", classes=1000, classifier_activation="softmax")

for i, layer in enumerate(VGG_19_base.layers):
    layer.trainable = False

print(VGG_19_base.summary())

# @title Layers para fine-tuning ---------------------------------------------------------------------------------------
FC_layer_Flatten = tf.keras.layers.Flatten()(VGG_19_base.output)
Dense = tf.keras.layers.Dense(units=200, activation="relu")(FC_layer_Flatten)
Dense = tf.keras.layers.Dense(units=160, activation="relu")(Dense)
Dense = tf.keras.layers.Dense(units=80, activation="relu")(Dense)
Dense = tf.keras.layers.Dense(units=40, activation="relu")(Dense)
Dense = tf.keras.layers.Dense(units=20, activation="relu")(Dense)
Classification = tf.keras.layers.Dense(units=2, activation="softmax")(Dense)

modelo_final = tf.keras.Model(inputs=image_input, outputs=Classification)
modelo_final.summary()

base_learning_rate = 0.001
epocas = 100

modelo_final.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                     loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

history = modelo_final.fit(x=train_ds, epochs=epocas, batch_size=32, validation_data=validation_ds)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Acurácia do Modelo")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.legend(['train', 'validation'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Perda do Modelo")
plt.xlabel("Época")
plt.ylabel("Perda")
plt.legend(['train', 'validation'])
plt.show()

prediction = modelo_final.predict(validation_ds)

shutil.rmtree("./training")
shutil.rmtree("./validation")
