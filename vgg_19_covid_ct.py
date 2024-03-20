# -*- coding: utf-8 -*-
"""VGG-19 Covid CT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sfvSv9RwAA_U7AZ9X-b_i9YplM-hWt5l

# Importações
"""

import tensorflow as tf

import os
import shutil
from matplotlib import pyplot as plt

import random

# @title Integração com o Kaggle
os.system('pip install kaggle >/dev/null')
kaggle_dir = os.path.expanduser('~/.kaggle')
if not os.path.exists(kaggle_dir):
    os.mkdir(kaggle_dir)

os.system('cp /VGG_19_Covid/kaggle.json ~/.kaggle/')
os.system('chmod 600 ~/.kaggle/kaggle.json')

"""# Preparação do Dataset"""

# @title Download a partir do Kaggle
'''
  Se o dataset ainda não foi baixado, baixamos e descompactamos
  Mantemos apenas as imagens originais, e não as pré-processadas (faremos o pré-processamento nós mesmos)
'''
if not os.path.exists("/VGG_19_Covid/dataset"):
    os.system('kaggle datasets download -d azaemon/preprocessed-ct-scans-for-covid19')
    print("Dataset download sucessful")
    os.system('unzip /VGG_19_Covid/preprocessed-ct-scans-for-covid19.zip -d /VGG_19_Covid/preprocessed-ct-scans-for-covid19')
    print("Dataset unziping sucessful")
    os.system('rm /VGG_19_Covid/preprocessed-ct-scans-for-covid19.zip')
    print("Zip file removal sucessful")
    os.rename("/VGG_19_Covid/preprocessed-ct-scans-for-covid19/", "/VGG_19_Covid/dataset")
    print("Dataset folder renamed sucessfuly")

    os.system('mv "/VGG_19_Covid/dataset/Original CT Scans/NiCT" "/VGG_19_Covid/dataset/Original CT Scans/nCT" "/VGG_19_Covid/dataset"')
    os.system('rm -r "/VGG_19_Covid/dataset/Original CT Scans"')

    dir = os.listdir("/VGG_19_Covid/dataset/NiCT")
    for i in dir:
        shutil.move(os.path.join("/VGG_19_Covid/dataset/NiCT", i), "/VGG_19_Covid/dataset/nCT/")

    os.rename("/VGG_19_Covid/dataset/nCT", "/VGG_19_Covid/dataset/negative")
    os.rename("/VGG_19_Covid/dataset/pCT", "/VGG_19_Covid/dataset/positive")

    os.makedirs("/VGG_19_Covid/dataset/training/positive/", exist_ok=True)
    os.makedirs("/VGG_19_Covid/dataset/training/negative/", exist_ok=True)
    os.makedirs("/VGG_19_Covid/dataset/validation/positive/", exist_ok=True)
    os.makedirs("/VGG_19_Covid/dataset/validation/negative/", exist_ok=True)


# @title Restrição do tamanho do dataset
# utilizando um sampling randomizado, separamos as imagens em conjuntos de treinamento e validação

sampling_training = 0.4
sampling_validation = 0.2

path = "/VGG_19_Covid/dataset/"

# selecionamos 30% das imagens positivas para treinamento
dir = os.listdir(path + "positive")
dir = random.sample(dir, (int)(len(dir) * sampling_training) )
source = path + "positive/"
for imagem in dir:
  shutil.move(source + imagem, path + "training/positive/")

#selecionamos 30% das imagens negativas para treinamento
dir = os.listdir(path + "negative")
dir = random.sample(dir, (int)(len(dir) * sampling_training) )
source = path + "negative/"
for imagem in dir:
  shutil.move(source + imagem, path + "training/negative/")


#selecionamos 10% das imagens positivas para validação
dir = os.listdir(path + "positive")
dir = random.sample(dir, (int)(len(dir) * sampling_validation) )
source = path + "positive/"
for imagem in dir:
  shutil.move(source + imagem, path + "validation/positive/")

#selecionamos 10% das imagens negativas para validação
dir = os.listdir(path + "negative")
dir = random.sample(dir, (int)(len(dir) * sampling_validation) )
source = path + "negative/"
for imagem in dir:
  shutil.move(source + imagem, path + "validation/negative/")

os.system('rm -r "/VGG_19_Covid/dataset/negative"')
os.system('rm -r "/VGG_19_Covid/dataset/positive"')

# @title Criação de dataset keras
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/VGG_19_Covid/dataset/training",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(224,224))

validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/VGG_19_Covid/dataset/validation",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(224,224))

"""# Aplicação da VGG-19"""

image_input = tf.keras.layers.Input(shape=(224,224, 3))
VGG_19_base = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_tensor=image_input, pooling="max", classes=1000, classifier_activation="softmax")

for i, layer in enumerate(VGG_19_base.layers):
  layer.trainable = False

print(VGG_19_base.summary())

# @title Layers para fine-tuning
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

modelo_final.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

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
