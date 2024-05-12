# Importações ----------------------------------------------------------------------------------------------------------
from datetime import datetime
from pathlib import Path
from random import seed, randint

import tensorflow as tf
from keras.callbacks import CSVLogger
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from functions import plot_images_from_dataset, preparar_dataset2, treinamento_e_teste, preparar_dataset1, \
    prepararDiretorios

seed()

# Parâmetros do treinamento --------------------------------------------------------------------------------------------
base_learning_rate = 0.001
epocas = 100

test_split = 0.2
validation_split = 0.4

dataset_dir1 = Path("./dataset1")  # sarscov2-ctscan-dataset
dataset_dir2 = Path("./dataset2")  # preprocessed-ct-scans-for-covid19

N = prepararDiretorios(dataset_dir1, dataset_dir2, test_split)

for i in [1, 2]:
    # Criação de datasets keras ----------------------------------------------------------------------------------------
    print("\nCriando datasets a partir dos diretórios...")

    train_ds, validation_ds = tf.keras.utils.image_dataset_from_directory(
        directory=f"./treinamento{i}",
        labels="inferred",
        label_mode="categorical",
        seed=randint(0, 100),
        subset="both",
        validation_split=validation_split,
        batch_size=32,
        image_size=(256, 256))

    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=f"./teste{i}",
        labels="inferred",
        label_mode="categorical",
        batch_size=32,
        image_size=(256, 256))

    print("Pronto!\n")

    # Aplicação da VGG-19 ----------------------------------------------------------------------------------------------
    # criar a base do modelo
    image_inputs2 = tf.keras.layers.Input(shape=(256, 256, 3))
    VGG_19_base = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_tensor=image_inputs2,
                                              pooling="max", classifier_activation="softmax")

    # congelar a base para transfer learning
    for _, layer in enumerate(VGG_19_base.layers):
        layer.trainable = False

    # layers para fine-tuning do transfer-learning
    FC_layer_Flatten = tf.keras.layers.Flatten()(VGG_19_base.output)
    x = tf.keras.layers.Dropout(0.3)(FC_layer_Flatten)
    x = tf.keras.layers.Dense(units=128, activation="relu")(x)
    x = tf.keras.layers.Dense(units=32, activation="relu")(x)
    x = tf.keras.layers.Dense(units=8, activation="relu")(x)
    outputs = tf.keras.layers.Dense(units=2, activation="softmax")(x)

    # compilar o modelo final
    modelo_final_transfer = tf.keras.Model(inputs=image_inputs2, outputs=outputs)
    modelo_final_transfer.summary()
    print("\n")

    modelo_final_transfer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                                  loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    # criar um log em CSV para armazenar a história do modelo
    csv_logger = CSVLogger(f"resultados/training{i}.csv", ",", False)

    # treinamento
    print(f"------------------------- Treinamento do modelo {i} -------------------------")
    start_time = datetime.now()
    history = modelo_final_transfer.fit(train_ds, validation_data=validation_ds, epochs=epocas, callbacks=[csv_logger],
                                        verbose=2)
    end_time = datetime.now()

    # teste
    print(f"\n---------------------------- Teste do modelo {i} ----------------------------")
    test_score = modelo_final_transfer.evaluate(test_ds)

    # Registro dos resultados ------------------------------------------------------------------------------------------
    print("Salvando informações e resultados...\n\n")

    # plotar uma imagem com exemplos de imagesn positivas e negativas
    plot_images_from_dataset(test_ds, f"resultados/sample{i}.png")

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Acurácia do Modelo")
    plt.xlabel("Épocas")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # garante que só inteiros serão utilizados na escala
    plt.ylabel("Acurácia")
    plt.legend(['train', 'validation'])
    plt.savefig(f"./resultados/acuracia{i}.png", bbox_inches='tight')
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Perda do Modelo")
    plt.xlabel("Época")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # garante que só inteiros serão utilizados na escala
    plt.ylabel("Perda")
    plt.legend(['train', 'validation'])
    plt.savefig(f"./resultados/perda{i}.png", bbox_inches='tight')
    plt.close()

    with open(f"resultados/info{i}.txt", "w") as file:
        file.write(f"N: {N[i]} imagens\n")
        file.write(f"Test split: {test_split}\n")
        file.write(f"Validation split: {validation_split}\n")
        file.write(f"Learning rate: {base_learning_rate}\n")
        file.write(f"Épocas: {epocas}\n")
        file.write(f"Tempo decorrido: {end_time - start_time}\n")
        file.write(f"Acurácia do deste: {test_score[1]}\n")
        file.write(f"Perda do deste: {test_score[0]}")

print("Pronto! Cheque a pasta resultados.")
