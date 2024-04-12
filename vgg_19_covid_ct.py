# -*- coding: utf-8 -*-

# Importações ----------------------------------------------------------------------------------------------------------
import os
import shutil
from pathlib import Path
from multiprocessing import Pool

from keras.callbacks import CSVLogger
import tensorflow as tf

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

import random
import datetime


# Funções auxiliares ---------------------------------------------------------------------------------------------------
def formatar_diretorio(origem, destino):
    """Move todos os arquivos de um diretório de origem para um destino e remove o diretório de origem.

    :param origem: Diretório de origem contendo os arquivos.
    :param destino: Diretório de destino.
    :return:
    """
    if not destino.exists():
        destino.mkdir()
    for file in origem.iterdir():
        shutil.move(file, destino / file.name)
    shutil.rmtree(origem)


def mover_imagens(origem, num_imagens, destino):
    """Move um número especificado de imagens de um diretório de origem para um destino.

    :param origem: Nome do diretório de origem contendo as imagens.
    :param num_imagens: O número de imagens para mover.
    :param destino: Nome do diretório de destino (incluindo subdiretórios para classes).
    :return:
    """
    destino_path = Path(destino)
    if not destino_path.exists():
        destino_path.mkdir(parents=True)
        (destino_path / "positivo").mkdir()
        (destino_path / "negative").mkdir()

    source_path = Path(origem)
    imagens = source_path.glob("*.jpg")

    for count, image_name in enumerate(imagens):
        if count >= num_imagens:
            break
        src = source_path / image_name.name
        dst = destino_path / image_name.name
        shutil.move(src, dst)


def plot_images_from_dataset(ds, filename):
    """Plota imagens de um tf.data.Dataset e salva a figura.

    :param ds: O número de imagens para plotar (padrão: 9).
    :param filename: Nome do arquivo para salvar a figura.
    :return:
    """
    imagens, labels = next(iter(ds.take(1)))
    label_indices = np.argmax(labels, axis=1)
    rows, cols = 3, 3
    plt.figure(figsize=(cols * 2.5, rows * 2.5))
    for i in range(min(9, len(imagens))):
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(imagens[i].numpy().astype("uint8"))
        plt.title(ds.class_names[label_indices[i]])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


random.seed(datetime.time.min.microsecond.__int__())

# Download a partir do Kaggle, e formatação do diretório ---------------------------------------------------------------
'''
  Se o dataset ainda não foi baixado, baixamos e descompactamos
  Mantemos apenas as imagens originais, e não as pré-processadas
'''
dataset_dir = Path("./dataset")

if not dataset_dir.exists():
    print("Baixando dataset de imagens e criando diretório...\n")
    dataset_dir.mkdir()

    os.system(f'kaggle datasets download -d azaemon/preprocessed-ct-scans-for-covid19 -p {dataset_dir} --unzip -q')
    shutil.rmtree(f"{dataset_dir}/Preprocessed CT scans")

    positive_dir = dataset_dir / "Original CT Scans/pCT"
    negative_dir = dataset_dir / "Original CT Scans/nCT"
    non_informative_dir = dataset_dir / "Original CT Scans/NiCT"

    formatar_diretorio(positive_dir, dataset_dir / "positive")
    formatar_diretorio(negative_dir, dataset_dir / "negative")
    formatar_diretorio(non_informative_dir, dataset_dir / "negative")

    shutil.rmtree(dataset_dir / "Original CT Scans")

    print("Pronto!")
else:
    print("Diretório de imagens já está presente na máquina. Prosseguindo...")

if not Path("resultados").exists():
    Path("resultados").mkdir()

# Criação de diretórios de treinamento e teste -------------------------------------------------------------------------
num_positivas = len(list(Path("./dataset/positive").glob("*.jpg")))
num_negativas = len(list(Path("./dataset/negative").glob("*.jpg")))

test_split = 0.2
validation_split = 0.4

if not (Path("treinamento").exists() & Path("./teste").exists()):
    print("\nCriando diretórios para treinamento e teste...")
    if not Path("./temp").exists():
        print("\tCriando diretório temporário...")
        Path("./temp").mkdir()
        with Pool() as pool:
            pool.apply_async(shutil.copytree, (f"./{dataset_dir}/positive", "./temp/positive"))
            pool.apply_async(shutil.copytree, (f"./{dataset_dir}/negative", "./temp/negative"))
            pool.close()
            pool.join()
        print("\tPronto!\nProsseguindo...")

    mover_imagens("./temp/positive", int(num_positivas * test_split), "./teste/positive")
    mover_imagens("./temp/negative", int(num_negativas * test_split), "./teste/negative")
    mover_imagens("./temp/positive", num_positivas - int(num_positivas * test_split), "./treinamento/positive")
    mover_imagens("./temp/negative", num_negativas - int(num_negativas * test_split), "./treinamento/negative")

    shutil.rmtree("./temp")
    print("Pronto!")
else:
    print("Diretórios de treinamento e teste já estão presentes. Prosseguindo...")

# Criação de datasets keras --------------------------------------------------------------------------------------------
print("\nCriando datasets a partir dos diretórios...")
train_ds, validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory="./treinamento",
    labels="inferred",
    label_mode="categorical",
    seed=random.randint(0, 100),
    subset="both",
    validation_split=validation_split,
    batch_size=32,
    image_size=(256, 256))

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory="./teste",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(256, 256))
print("Pronto!\n")

# Aplicação da VGG-19 --------------------------------------------------------------------------------------------------
image_inputs = tf.keras.layers.Input(shape=(256, 256, 3))
VGG_19_base = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_tensor=image_inputs,
                                          pooling="max", classifier_activation="softmax")

for i, layer in enumerate(VGG_19_base.layers):
    layer.trainable = False

# Layers para fine-tuning
FC_layer_Flatten = tf.keras.layers.Flatten()(VGG_19_base.output)
x = tf.keras.layers.Dropout(0.3)(FC_layer_Flatten)
x = tf.keras.layers.Dense(units=128, activation="relu")(x)
x = tf.keras.layers.Dense(units=32, activation="relu")(x)
x = tf.keras.layers.Dense(units=8, activation="relu")(x)
outputs = tf.keras.layers.Dense(units=2, activation="softmax")(x)

# Compilação do modelo final
modelo_final = tf.keras.Model(inputs=image_inputs, outputs=outputs)
modelo_final.summary()
print("\n")

base_learning_rate = 0.001
epocas = 50

modelo_final.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                     loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

# cria um log em CSV para armazenar a história do modelo
csv_logger = CSVLogger("resultados/training.csv", ",", False)

# Treinamento
print("----------------------- Treinamento do modelo -----------------------")
start_time = datetime.datetime.now()
history = modelo_final.fit(train_ds, validation_data=validation_ds, epochs=epocas, callbacks=[csv_logger], verbose=2)
end_time = datetime.datetime.now()

# Teste
print("\n-------------------------- Teste do modelo --------------------------")
test_score = modelo_final.evaluate(test_ds)

# Registro dos resultados ----------------------------------------------------------------------------------------------
print("Salvando informações e resultados...")

plot_images_from_dataset(test_ds, "resultados/sample.png")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Acurácia do Modelo")
plt.xlabel("Épocas")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # garante que só inteiros serão utilizados na escala
plt.ylabel("Acurácia")
plt.legend(['train', 'validation'])
plt.savefig("./resultados/acuracia.png", bbox_inches='tight')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Perda do Modelo")
plt.xlabel("Época")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # garante que só inteiros serão utilizados na escala
plt.ylabel("Perda")
plt.legend(['train', 'validation'])
plt.savefig("./resultados/perda.png", bbox_inches='tight')
plt.close()

with open("resultados/info.txt", "w") as file:
    file.write(f"Test split: {test_split}\n")
    file.write(f"Validation split: {validation_split}\n")
    file.write(f"Learning rate: {base_learning_rate}\n")
    file.write(f"Épocas: {epocas}\n")
    file.write(f"Tempo decorrido: {end_time - start_time}\n")
    file.write(f"Acurácia do deste: {test_score[1]}\n")
    file.write(f"Perda do deste: {test_score[0]}")

print("Pronto! Cheque a pasta resultados.")
