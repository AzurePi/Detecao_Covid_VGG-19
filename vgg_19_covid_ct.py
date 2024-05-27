# Importações ----------------------------------------------------------------------------------------------------------
from datetime import datetime
from pathlib import Path
from random import seed, randint

import tensorflow as tf
from keras.callbacks import CSVLogger

from functions import salvar_parametros, prepararResultados, prepararDiretorios, plotar_amostra, plotar_graficos, \
    salvar_resultados

# Parâmetros do treinamento ----------------------------------------------------------------------------------------
base_learning_rate = 0.001
epocas = 2
test_split = 0.2
validation_split = 0.4
dataset_dir1 = Path("./dataset1")  # sarscov2-ctscan-dataset
dataset_dir2 = Path("./dataset2")  # preprocessed-ct-scans-for-covid19


def main():
    seed()

    salvar_parametros(base_learning_rate, epocas, test_split, validation_split)

    training, n = prepararResultados()

    histories, histories_transfer = [], []
    test_scores, test_scores_transfer = [], []

    loss_max = float('-inf')

    N = prepararDiretorios(dataset_dir1, dataset_dir2, test_split)

    for i in [1, 2]:
        print(f"\nCriando datasets keras a partir do dataset {i}...")

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

        plotar_amostra(test_ds, f"resultados/sample{i}.png")

        print("Pronto!\n")

        # Aplicação da VGG-19 ------------------------------------------------------------------------------------------
        # criar a base do modelo
        image_inputs = tf.keras.layers.Input(shape=(256, 256, 3))
        VGG_19_base = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_tensor=image_inputs,
                                                  pooling="max", classifier_activation="softmax")

        # compilamos um modelo para treinamento completo
        outputs = tf.keras.layers.Dense(units=2, activation="softmax")(VGG_19_base.output)
        modelo_final = tf.keras.Model(inputs=image_inputs, outputs=outputs)
        print("Modelo:")
        modelo_final.summary()
        print("\n")

        modelo_final.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                             loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

        # congelamos a base para transfer learning
        for _, layer in enumerate(VGG_19_base.layers):
            layer.trainable = False

        # adicionamos alguns layers para fine-tuning do transfer-learning
        FC_layer_Flatten = tf.keras.layers.Flatten()(VGG_19_base.output)
        x = tf.keras.layers.Dropout(0.3)(FC_layer_Flatten)
        x = tf.keras.layers.Dense(units=128, activation="relu")(x)
        x = tf.keras.layers.Dense(units=32, activation="relu")(x)
        x = tf.keras.layers.Dense(units=8, activation="relu")(x)
        outputs = tf.keras.layers.Dense(units=2, activation="softmax")(x)

        # compilamos um modelo para treinamento com transfer learning
        modelo_final_transfer = tf.keras.Model(inputs=image_inputs, outputs=outputs)
        print("Modelo com transfer learning:")
        modelo_final_transfer.summary()
        print("\n")

        modelo_final_transfer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                                      loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

        # criamos logs em CSV para armazenar as histórias de cada modelo
        csv_logger = CSVLogger(training / f"normal/dataset{i}_({n}).csv", ",", False)
        csv_logger_transfer = CSVLogger(training / f"transfer/dataset{i}_({n}).csv", ",", False)

        print(f"------------------------- Treinamento com dataset {i} -------------------------")
        print("Normal:")
        start_time = datetime.now()
        history = modelo_final.fit(train_ds, validation_data=validation_ds, epochs=epocas,
                                   callbacks=[csv_logger],
                                   verbose=2)
        end_time = datetime.now()
        time = end_time - start_time

        print("\nCom transfer learning:")
        start_time = datetime.now()
        history_transfer = modelo_final_transfer.fit(train_ds, validation_data=validation_ds, epochs=epocas,
                                                     callbacks=[csv_logger_transfer],
                                                     verbose=2)
        end_time = datetime.now()
        time_transfer = end_time - start_time

        print(f"\n---------------------------- Teste do modelo {i} ----------------------------")
        print("Normal:")
        test_score = modelo_final.evaluate(test_ds)
        print("\nCom transfer learning:")
        test_score_transfer = modelo_final_transfer.evaluate(test_ds)

        # Registro dos resultados --------------------------------------------------------------------------------------
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
    for i in [1, 2]:
        plotar_graficos(epocas, histories[i - 1], test_scores[i - 1], histories_transfer[i - 1],
                        test_scores_transfer[i - 1], i, loss_max)
    print("Pronto! Cheque a pasta resultados.")


main()
