# Importações ----------------------------------------------------------------------------------------------------------
from datetime import datetime
from pathlib import Path

import tensorflow as tf
import keras

from datasets import carregar_datasets, preparar_diretorios
from resultados import salvar_parametros, prepararResultados, plotar_amostra, salvar_resultados, plotar_graficos

# Parâmetros do treinamento --------------------------------------------------------------------------------------------
base_learning_rate = 1e-4
batch_size = 32
epocas = 100
test_split = 0.2
validation_split = 0.4

dataset_dir1 = Path("./dataset1")  # sarscov2-ctscan-dataset
dataset_dir2 = Path("./dataset2")  # preprocessed-ct-scans-for-covid19
dataset_dir3 = Path("./dataset3")  # combinação dos outros dois datasets

metrics = ['accuracy']


# Funções Auxiliares ---------------------------------------------------------------------------------------------------

def criar_modelo(pesos, congelar_base) -> tf.keras.Model:
    """
    Cria e compila um modelo VGG-19 para treinamento, inicializado com pesos especificados.

    Utiliza pesos especificados (pré-treinados ou aleatórios) e configura o modelo para transfer learning se
    `congelar_base` for True. O modelo é compilado com o otimizador `Adam`, e a função de perda `CategoricalCrossentropy`.

    :return: Um modelo VGG-19 compilado, pronto para treinamento.
    """
    image_inputs = tf.keras.layers.Input(shape=(256, 256, 3))
    VGG_19 = tf.keras.applications.VGG19(include_top=False, weights=pesos, input_tensor=image_inputs, pooling="max")

    if congelar_base:
        for layer in VGG_19.layers:
            layer.trainable = False

    x = tf.keras.layers.Flatten()(VGG_19.output)
    x = tf.keras.layers.Dense(units=4096, activation="relu")(x)
    x = tf.keras.layers.Dense(units=4096, activation="relu")(x)
    outputs = tf.keras.layers.Dense(units=2, activation="softmax", name="predictions")(x)

    modelo = tf.keras.Model(inputs=image_inputs, outputs=outputs)
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                   loss=tf.keras.losses.CategoricalCrossentropy(), metrics=metrics)
    return modelo


def criar_modelo_normal() -> tf.keras.Model:
    """
    Cria e compila um modelo VGG-19 com pesos inicializados aleatoriamente.

    :return: Um modelo VGG-19 compilado com pesos aleatórios.
    """
    return criar_modelo(pesos=None, congelar_base=False)


def criar_modelo_transfer() -> tf.keras.Model:
    """
    Cria e compila um modelo VGG-19 usando pesos pré-treinados no ImageNet, com a base congelada para transfer learning.

    :return: Um modelo VGG-19 compilado com pesos pré-treinados no ImageNet e a base congelada.
    """
    return criar_modelo(pesos="imagenet", congelar_base=True)


class CallbackAcuraciaExcedida(tf.keras.callbacks.Callback):
    def __init__(self):
        super(CallbackAcuraciaExcedida, self).__init__()
        self.epoch_acuracia_excedida = None

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') >= 0.9 and self.epoch_acuracia_excedida is None:
            self.epoch_acuracia_excedida = epoch


# Main -----------------------------------------------------------------------------------------------------------------

def main():
    training, n = prepararResultados()
    salvar_parametros(base_learning_rate, batch_size, epocas, test_split, validation_split)

    # variáveis para armazenar o desempenho dos modelos em cada dataset
    histories, histories_transfer = dict(), dict()
    test_scores, test_scores_transfer = dict(), dict()
    test_scores_others, test_scores_transfer_others = dict(), dict()
    epocas_acuracia_exc, epocas_acuracia_exc_transfer = dict(), dict()

    loss_max = float('-inf')
    N = preparar_diretorios(test_split, [dataset_dir1, dataset_dir2, dataset_dir3])

    # Criação dos modelos (salvamos os pesos de inicialização para garantir consistência a cada treinamento)
    modelo = criar_modelo_normal()
    print("Normal:")
    modelo.summary()
    initial_weights = modelo.get_weights()

    modelo_transfer = criar_modelo_transfer()
    print("\nCom transfer learning:")
    modelo_transfer.summary()
    initial_weights_transfer = modelo_transfer.get_weights()
    print("\n")

    # usamos um callback personalizado, que salva a época do treinamento em que a acurácia de validação passou 0.9
    callback_acuracia_excedida = CallbackAcuraciaExcedida()

    for i in [1, 2, 3]:
        print(f"\nCriando datasets keras a partir do diretório dataset{i}...")
        train_ds, validation_ds, test_ds, class_names = carregar_datasets(i, validation_split, batch_size)
        plotar_amostra(train_ds, f"resultados/sample{i}.png", class_names)

        print("Pronto!\n")

        # criamos logs em CSV para armazenar as histórias de cada modelo
        csv_logger = keras.callbacks.CSVLogger(training / f"normal/dataset{i} ({n}).csv", ",", False)
        csv_logger_transfer = keras.callbacks.CSVLogger(training / f"transfer/dataset{i} ({n}).csv", ",", False)

        # inicializamos os modelos com os mesmos pesos a cada iteração
        modelo.set_weights(initial_weights)
        modelo_transfer.set_weights(initial_weights_transfer)

        # Treinamento e testes -----------------------------------------------------------------------------------------
        print(f"------------------------------------ Treinamentos com dataset {i} ------------------------------------")

        # Treinamento normal
        print("Normal:")
        start_time = datetime.now()
        history = modelo.fit(train_ds, validation_data=validation_ds, epochs=epocas,
                             callbacks=[csv_logger, callback_acuracia_excedida], verbose=1)
        time = datetime.now() - start_time
        epoca_acuracia_exc = callback_acuracia_excedida.epoch_acuracia_excedida

        # Treinamento com transfer learning
        print("\nCom transfer learning:")
        start_time = datetime.now()
        history_transfer = modelo_transfer.fit(train_ds, validation_data=validation_ds, epochs=epocas,
                                               callbacks=[csv_logger_transfer, callback_acuracia_excedida], verbose=1)
        time_transfer = datetime.now() - start_time
        epoca_acuracia_exc_transfer = callback_acuracia_excedida.epoch_acuracia_excedida

        print(f"\n--------------------------------------- Testes do modelo {i} ---------------------------------------")

        # Testes com o mesmo dataset
        print(f"Testes com o dataset {i}:\n")
        test_score = modelo.evaluate(test_ds)
        test_score_transfer = modelo_transfer.evaluate(test_ds)

        # Testes com os outros datasets
        print("\nRealizando testes com os demais datasets:\n")
        test_score_other, test_score_transfer_other = dict(), dict()
        for j in [x for x in [1, 2, 3] if x != i]:
            ds = f"dataset{j}"
            _, _, test_ds_other, _ = carregar_datasets(j, validation_split, batch_size)
            test_score_other[ds] = modelo.evaluate(test_ds_other)  # Teste normal
            test_score_transfer_other[ds] = modelo_transfer.evaluate(test_ds_other)  # Teste com transfer learning
            print("\n")

        # Registro dos resultados --------------------------------------------------------------------------------------
        print(f"Salvando resultados para o dataset {i}...\n\n")
        ds = f"dataset{i}"
        histories[ds] = history
        histories_transfer[ds] = history_transfer
        test_scores[ds] = test_score
        test_scores_transfer[ds] = test_score_transfer
        test_scores_others[ds] = test_score_other
        test_scores_transfer_others[ds] = test_score_transfer_other
        epocas_acuracia_exc[ds] = epoca_acuracia_exc
        epocas_acuracia_exc_transfer[ds] = epoca_acuracia_exc_transfer

        salvar_resultados(N, i, test_score, test_score_transfer, time, time_transfer, test_score_other,
                          test_score_transfer_other, epoca_acuracia_exc, epoca_acuracia_exc_transfer)

        # lista para armazenar todos os valores de loss para simplificar a cálulo da máxima
        loss_values = [
            loss_max,
            *history.history['loss'], *history.history['val_loss'],
            *history_transfer.history['loss'], *history_transfer.history['val_loss'],
            test_score[0],
            test_score_transfer[0],
            *[test_score_other[score][0] for score in test_score_other],
            *[test_score_transfer_other[score][0] for score in test_score_transfer_other]
        ]
        loss_max = max(loss_values)

    print("Plotando gráficos dos resultados...")
    for i in [1, 2, 3]:
        ds = f"dataset{i}"
        plotar_graficos(i=i, n=n, loss_max=loss_max,
                        history=histories[ds], history_transfer=histories_transfer[ds],
                        epoca_acuracia_excede=epocas_acuracia_exc[ds],
                        epoca_acuracia_excede_transfer=epocas_acuracia_exc_transfer[ds],
                        test=test_scores[ds], test_transfer=test_scores_transfer[ds],
                        test_others=test_scores_others[ds], test_transfer_others=test_scores_transfer_others[ds], )
    print("Pronto! Cheque a pasta resultados.")


if __name__ == "__main__":
    main()
