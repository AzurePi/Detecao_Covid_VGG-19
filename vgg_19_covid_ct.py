# -*- coding: utf-8 -*-

# Importações ----------------------------------------------------------------------------------------------------------

import os
import shutil

from keras.callbacks import CSVLogger
import tensorflow as tf

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

import random
import datetime


# Funções auxiliares ---------------------------------------------------------------------------------------------------
def plot_images_from_dataset(ds, filename):
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        num_images = min(len(images), 9)
        label_indices = np.argmax(labels, axis=1)
        for i in range(num_images):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(ds.class_names[label_indices[i]])
            plt.axis("off")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


random.seed(datetime.time.min.microsecond.__int__())

# Download a partir do Kaggle, pré-processamento e formatação do diretório ---------------------------------------------
'''
  Se o dataset ainda não foi baixado, baixamos e descompactamos
  Mantemos apenas as imagens originais, e não as pré-processadas
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
        shutil.move(src, dest)
    os.rmdir("./dataset/Original CT Scans")

    for file_name in os.listdir("./dataset/NiCT"):
        src = os.path.join("./dataset/NiCT", file_name)
        dest = os.path.join("./dataset/nCT", file_name)
        shutil.move(src, dest)
    os.rmdir("./dataset/NiCT")

    os.rename("./dataset/nCT", "./dataset/negative")
    os.rename("./dataset/pCT", "./dataset/positive")

if not (os.path.exists("./results")):
    os.mkdir("./results")

# Restrição do tamanho do dataset --------------------------------------------------------------------------------------
num_positivas = len(os.listdir("./dataset/positive"))
num_negativas = len(os.listdir("./dataset/negative"))

test_split = 0.2
validation_split = 0.4

# separamos as imagens em pastas
if not (os.path.exists("./training") & os.path.exists("./teste")):
    os.mkdir("./temp")
    shutil.copytree("./dataset", "./temp", dirs_exist_ok=True)

    if not (os.path.exists("./training")):
        os.mkdir("./training")
        os.mkdir("./training/positive")
        os.mkdir("./training/negative")

        diretorio = os.listdir("./temp/positive")
        random.shuffle(diretorio)
        source = "./temp/positive/"
        for imagem in diretorio[:(num_positivas * (1 - test_split)).__int__()]:
            shutil.move(f'{source}{imagem}', "./training/positive/")

        diretorio = os.listdir("./temp/negative")
        random.shuffle(diretorio)
        source = "./temp/negative/"
        for imagem in diretorio[:(num_negativas * (1 - test_split)).__int__()]:
            shutil.move(f'{source}{imagem}', "./training/negative/")

    if not (os.path.exists("./teste")):
        os.mkdir("./teste")
        os.mkdir("./teste/positive")
        os.mkdir("./teste/negative")

        diretorio = os.listdir("./temp/positive")
        random.shuffle(diretorio)
        source = "./temp/positive/"
        for imagem in diretorio[:(num_positivas * test_split).__int__()]:
            shutil.move(f'{source}{imagem}', "./teste/positive/")

        # selecionamos uma fração das imagens negativas para validação
        diretorio = os.listdir("./temp/negative")
        random.shuffle(diretorio)
        source = "./temp/negative/"
        for imagem in diretorio[:(num_negativas * test_split).__int__()]:
            shutil.move(f'{source}{imagem}', "./teste/negative/")

    shutil.rmtree("./temp")

    '''
        if not (os.path.exists("./validation")):
            os.mkdir("./validation")
            os.mkdir("./validation/positive")
            os.mkdir("./validation/negative")

            # selecionamos uma fração das imagens positivas para validação
            diretorio = os.listdir("./temp/positive")
            random.shuffle(diretorio)
            source = "./temp/positive/"
            for imagem in diretorio[:num_positivas // 5]:
                shutil.move(f'{source}{imagem}', "./validation/positive/")

            # selecionamos uma fração das imagens negativas para validação
            diretorio = os.listdir("./temp/negative")
            random.shuffle(diretorio)
            source = "./temp/negative/"
            for imagem in diretorio[:num_negativas // 5]:
                shutil.move(f'{source}{imagem}', "./validation/negative/")
    '''

# Criação de datasets keras --------------------------------------------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory="./training",
    labels="inferred",
    label_mode="categorical",
    seed=random.randint(0, 10000),
    subset="both",
    validation_split=validation_split,
    batch_size=32,
    image_size=(256, 256))

'''
validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory="./validation",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(256, 256))
plot_images_from_dataset(validation_ds, "./results/sampleValidation.png")
'''
test_ds = tf.keras.utils.image_dataset_from_directory(
    directory="./teste",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(256, 256))
plot_images_from_dataset(test_ds, "./results/sample.png")

# Aplicação da VGG-19 --------------------------------------------------------------------------------------------------
image_input = tf.keras.layers.Input(shape=(256, 256, 3))
VGG_19_base = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_tensor=image_input,
                                          pooling="max", classifier_activation="softmax")

for i, layer in enumerate(VGG_19_base.layers):
    layer.trainable = False

print(VGG_19_base.summary())

# Layers para fine-tuning
FC_layer_Flatten = tf.keras.layers.Flatten()(VGG_19_base.output)
Dense = tf.keras.layers.Dense(units=256, activation="relu")(FC_layer_Flatten)
Dense = tf.keras.layers.Dense(units=128, activation="relu")(Dense)
Dense = tf.keras.layers.Dense(units=64, activation="relu")(Dense)
Dense = tf.keras.layers.Dense(units=16, activation="relu")(Dense)
Dense = tf.keras.layers.Dense(units=4, activation="relu")(Dense)
Classification = tf.keras.layers.Dense(units=2, activation="softmax")(Dense)

# Compilação do modelo final
modelo_final = tf.keras.Model(inputs=image_input, outputs=Classification)
modelo_final.summary()

base_learning_rate = 0.001
epocas = 50

modelo_final.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                     loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

csv_logger = CSVLogger("./results/training.csv", ",", False)  # cria um log em CSV para armazenar a história do modelo

# Treinamento
start_time = datetime.datetime.now()
history = modelo_final.fit(x=train_ds[0], validation_data=train_ds[1], epochs=epocas, callbacks=[csv_logger], verbose=2)
end_time = datetime.datetime.now()

# Teste
test_score = modelo_final.evaluate(test_ds)

# Registro dos resultados ----------------------------------------------------------------------------------------------
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Acurácia do Modelo")
plt.xlabel("Épocas")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # garante que só inteiros serão utilizados na escala
plt.ylabel("Acurácia")
plt.legend(['train', 'validation'])
plt.savefig("./results/acuracia.png", bbox_inches='tight')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Perda do Modelo")
plt.xlabel("Época")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # garante que só inteiros serão utilizados na escala
plt.ylabel("Perda")
plt.legend(['train', 'validation'])
plt.savefig("./results/perda.png", bbox_inches='tight')
plt.close()

with open("results/info.txt", "w") as file:
    file.write(f"Test split: {test_split}\n")
    file.write(f"Validation split: {validation_split}\n")
    file.write(f"Learning rate: {base_learning_rate}\n")
    file.write(f"Épocas: {epocas}\n")
    file.write(f"Tempo decorrido: {end_time - start_time}\n")
    file.write(f"Acurácia do deste: {test_score[1]}\n")
    file.write(f"Perda do deste: {test_score[0]}")
