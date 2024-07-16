# Importações ----------------------------------------------------------------------------------------------------------
from datetime import datetime
from pathlib import Path
from random import seed, randint

import tensorflow as tf
from keras.callbacks import CSVLogger

from functions import salvar_parametros, prepararResultados, prepararDiretorios, plotar_amostra, salvar_resultados, \
    plotar_graficos

# Parâmetros do treinamento --------------------------------------------------------------------------------------------
base_learning_rate = 0.0001
epocas = 100
test_split = 0.3
validation_split = 0.5

dataset_dir1 = Path("./dataset1")  # sarscov2-ctscan-dataset
dataset_dir2 = Path("./dataset2")  # preprocessed-ct-scans-for-covid19
dataset_dir3 = Path("./dataset3")  # combinação dos outros dois datasets

metrics = ['accuracy']


def carregar_datasets(i):
    seed = randint(0, 100)
    labels, label_mode = "inferred", "categorical"
    batch_size = 32
    image_size = (256, 256)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=f"./treinamento{i}",
        labels=labels,
        label_mode=label_mode,
        seed=seed,
        subset="training",
        validation_split=validation_split,
        batch_size=batch_size,
        image_size=image_size
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        directory=f"./treinamento{i}",
        labels=labels,
        label_mode=label_mode,
        seed=seed,
        subset="validation",
        validation_split=validation_split,
        batch_size=batch_size,
        image_size=image_size
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=f"./teste{i}",
        labels=labels,
        label_mode=label_mode,
        batch_size=batch_size,
        image_size=image_size
    )

    return train_ds, validation_ds, test_ds, train_ds.class_names


def criar_modelo(pesos=None, congelar_base=False):
    """
    Compila um modelo VGG-19 para treinamento, inicializado com pesos especificados. Se congelar_base = True, então o modelo será preparado para transfer-learning.
    """
    image_inputs = tf.keras.layers.Input(shape=(256, 256, 3))
    VGG_19 = tf.keras.applications.VGG19(include_top=False, weights=pesos,
                                         input_tensor=image_inputs, pooling="max")

    if congelar_base:
        for layer in VGG_19.layers:
            layer.trainable = False

    x = tf.keras.layers.Flatten()(VGG_19.output)
    x = tf.keras.layers.Dense(units=128, activation="relu")(x)
    x = tf.keras.layers.Dense(units=32, activation="relu")(x)
    x = tf.keras.layers.Dense(units=8, activation="relu")(x)
    outputs = tf.keras.layers.Dense(units=2, activation="softmax")(x)

    modelo = tf.keras.Model(inputs=image_inputs, outputs=outputs)
    modelo.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
                   loss=tf.keras.losses.CategoricalCrossentropy(), metrics=metrics)
    return modelo


def criar_modelo_normal():
    return criar_modelo(pesos=None, congelar_base=False)


def criar_modelo_transfer():
    return criar_modelo(pesos="imagenet", congelar_base=True)


def main():
    seed()

    training, n = prepararResultados()
    salvar_parametros(base_learning_rate, epocas, test_split, validation_split)

    histories, histories_transfer = [], []
    test_scores, test_scores_transfer = [], []

    loss_max = float('-inf')
    N = prepararDiretorios(test_split, [dataset_dir1, dataset_dir2, dataset_dir3])

    for i in [1, 2, 3]:
        print(f"\nCriando datasets keras a partir do diretório dataset{i}...")
        train_ds, validation_ds, test_ds, class_names = carregar_datasets(i)
        plotar_amostra(train_ds, f"resultados/sample{i}.png", class_names)

        print("Pronto!\n")

        # Criação dos modelos
        modelo = criar_modelo_normal()
        modelo_transfer = criar_modelo_transfer()

        print("Normal:")
        modelo.summary()
        print("\nCom transfer learning:")
        modelo_transfer.summary()
        print("\n")

        # criamos logs em CSV para armazenar as histórias de cada modelo
        csv_logger = CSVLogger(training / f"normal/dataset{i}_({n}).csv", ",", False)
        csv_logger_transfer = CSVLogger(training / f"transfer/dataset{i}_({n}).csv", ",", False)

        # Treinamento e teste ------------------------------------------------------------------------------------------
        print(f"------------------------- Treinamento com dataset {i} -------------------------")
        print("Normal:")
        start_time = datetime.now()
        history = modelo.fit(train_ds, validation_data=validation_ds, epochs=epocas,
                             callbacks=[csv_logger], verbose=2)
        time = datetime.now() - start_time

        print("\nCom transfer learning:")
        start_time = datetime.now()
        history_transfer = modelo_transfer.fit(train_ds, validation_data=validation_ds, epochs=epocas,
                                               callbacks=[csv_logger_transfer], verbose=2)
        time_transfer = datetime.now() - start_time

        # Teste
        print(f"\n---------------------------- Teste do modelo {i} ----------------------------")
        test_score = modelo.evaluate(test_ds)
        test_score_transfer = modelo_transfer.evaluate(test_ds)

        # Registro dos resultados
        print(f"\nSalvando resultados para o dataset {i}...\n\n")
        histories.append(history)
        histories_transfer.append(history_transfer)
        test_scores.append(test_score)
        test_scores_transfer.append(test_score_transfer)

        loss_max = max(loss_max, max(history.history['loss']), max(history.history['val_loss']),
                       max(history_transfer.history['loss']), max(history_transfer.history['val_loss']),
                       test_score[0], test_score_transfer[0])

        salvar_resultados(N, i, test_score, test_score_transfer, time, time_transfer)

    print("Plotando gráficos dos resultados...")
    for i in [1, 2, 3]:
        plotar_graficos(n, histories[i - 1], histories_transfer[i - 1], i, loss_max)
    print("Pronto! Cheque a pasta resultados.")

    # TODO: adicionar a sensibilidade


if __name__ == "__main__":
    main()
