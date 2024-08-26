from datetime import timedelta
from pathlib import Path

import keras.callbacks
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator


def salvar_parametros(base_learning_rate: float, batch_size: int, epocas: int, test_split: float,
                      validation_split: float) -> None:
    """
    Salva os parâmetros do treinamento em um arquivo "resultados/parametros.txt".

    :param base_learning_rate: Taxa de aprendizado utilizada no treinamento.
    :param batch_size: Tamanho das batches de imagens
    :param epocas: Número de épocas do treinamento.
    :param test_split: Proporção dos dados utilizados para teste.
    :param validation_split: Proporção dos dados utilizados para validação.
    :return: None
    """
    with open("./resultados/parametros.txt", 'w') as file:
        file.write(f"Learning rate: {base_learning_rate}\n")
        file.write(f"batch size: {batch_size}\n")
        file.write(f"Épocas: {epocas}\n")
        file.write(f"Test split: {test_split}\n")
        file.write(f"Validation split: {validation_split}\n")


def prepararResultados(metrics: list[str]) -> int:
    """
    Prepara a estrutura de diretórios para armazenar os resultados do treinamento. Calcula o número de treinamentos
    já feitos com base na quantidade de arquivos em "./resultados/logs/normal".

    Verifica se os diretórios "./resultados", "./resultados/logs", "./resultados/logs/normal" e
    "./resultados/logs/transfer" existem. Se não existirem, cria esses diretórios.

    :param: metrics: Lista com os nomes das métricas a serem salvas
    :return: Uma tupla contendo o caminho do diretório de treinamento e o número de treinamentos já feitos (calculado
    com base no número de logs encontrados)
    """
    dirs = ["./resultados/graphs", "./resultados/tempo", "./resultados/testes",
            "./resultados/logs/dataset1", "./resultados/logs/dataset2"]

    print("Preparando estrutura do diretório de resultados...\n")

    for path in dirs:
        Path(path).mkdir(parents=True, exist_ok=True)

    header = ';'.join(metrics) + ';' + ','.join([str(i) + "_transfer" for i in metrics])

    for i in [1, 2]:
        csv_files = [
            (f"./resultados/tempo/tempo{i}.csv", "tempo,tempo_transfer\n"),
            (f"./resultados/testes/testes{i}.csv", f"{header}\n"),
            (f"./resultados/testes/others{i}.csv", f"{header}\n")
        ]

        for p, m in csv_files:
            if not Path(p).exists():
                with open(p, "w") as file:
                    file.write(m)

    n = sum(1 for _ in Path("./resultados/logs/dataset1").iterdir()) // 2
    return n


def plotar_amostra(ds: tf.data.Dataset, filename: str, class_names: list[str]) -> None:
    """
    Plota 9 imagens de um dataset e salva a figura.

    :param ds: Dataset de onde tirar as imagens.
    :param filename: Nome do arquivo para salvar a figura.
    :param class_names: Nomes das classes às quais as imagens podem pertencer
    """
    rows, cols = 3, 3
    num_samples = rows * cols

    # calculamos o número de imagens por classe no dataset
    class_counts = {class_name: 0 for class_name in class_names}
    for image, labels in ds:
        for label in labels:
            class_name = class_names[label.numpy().argmax()]
            class_counts[class_name] += 1

    # calculamos a proporção
    total_count = sum(class_counts.values())
    class_proportions = {class_name: count / total_count for class_name, count in class_counts.items()}

    samples_per_class = {class_name: int(num_samples * class_proportion) for class_name, class_proportion in
                         class_proportions.items()}

    # garantimos que a o soma dos samples seja igual a num-samples
    remaining_samples = num_samples - sum(samples_per_class.values())
    for class_name in sorted(class_proportions, key=class_proportions.get, reverse=True):
        if remaining_samples == 0:
            break
        samples_per_class[class_name] += 1
        remaining_samples -= 1

    # coletamos as imagens proporcionalmente às classes
    collected_images = []
    collected_labels = []
    current_counts = {class_name: 0 for class_name in class_names}

    for images, labels in ds:
        for i in range(len(images)):
            label = labels[i].numpy().argmax()
            class_name = class_names[label]
            if current_counts[class_name] < samples_per_class[class_name]:
                collected_images.append(images[i])
                collected_labels.append(labels[i])
                current_counts[class_name] += 1
            if len(collected_images) == num_samples:
                break
        if len(collected_images) == num_samples:
            break

    plt.figure(figsize=(cols * 2.5, rows * 2.5))
    for i in range(num_samples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(collected_images[i].numpy().astype("uint8"))
        plt.title(class_names[collected_labels[i].numpy().argmax()])
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def grafico_aux(i: int, titulo: str, treino, validacao, teste, teste_other, ylabel: str, output_path: Path) -> None:
    """
    Função auxiliar para plotar gráficos de perda e acurácia.

    :param i: Número do dataset.
    :param titulo: Título do gráfico.
    :param treino: Dados de treino.
    :param validacao: Dados de validação.
    :param teste: Dados de teste.
    :param teste_other: Dados de teste com os demais datasets.
    :param ylabel: Rótulo do eixo y.
    :param output_path: Caminho para salvar a imagem.
    :return: None
    """
    plt.title(titulo)
    plt.plot(treino, label="Treino")
    plt.plot(validacao, label="Validação")

    plt.xlabel("Época")
    plt.xticks(np.arange(0, len(treino), step=len(treino) // 5))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # garante que só inteiros serão utilizados na escala

    plt.ylabel(ylabel)

    # Ajuste dinâmico da escala do gráfico
    if ylabel == "Perda":
        # Definindo o limite dos eixos para a perda com base nos valores de treino e validação
        min_loss = min(min(treino), min(validacao))
        max_loss = max(max(treino), max(validacao))

        plt.ylim(min_loss - 0.1 * abs(min_loss), max_loss + 0.1 * abs(max_loss))

        # Calculando uma escala apropriada para o minor locator
        m = (max_loss - min_loss) / 10
        plt.gca().yaxis.set_minor_locator(MultipleLocator(m))
    else:  # Para a acurácia
        plt.gca().yaxis.set_minor_locator(MultipleLocator(0.05))
        if i == 2:  # Limitar o gráfico entre 0.7 e 1.0 para a acurácia do dataset 2
            plt.ylim(0.7, 1.0)
        else:
            plt.ylim(0, 1.0)  #
        j = 3 - i
        plt.plot(len(treino) - 1, teste['accuracy'], 'o', label=f"Teste (dataset {i})")
        plt.plot(len(treino) - 1, teste_other[f"dataset{j}"]['accuracy'], 'o', linewidth=0.5,
                 label=f"Teste (dataset {j})")

    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.legend(shadow=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plotar_graficos(i: int, n: int, history: keras.callbacks.History, history_transfer: keras.callbacks.History, test,
                    test_transfer, test_others, test_transfer_others) -> None:
    """
    Plota os gráficos de acurácia e perda dos modelos treinados, com e sem transfer learning. Separa os gráficos em
    pastas diferentes segundo o dataset utilizado.


    Garante que os eixos Y sejam os mesmos para todos os gráficos de acurácia e perda.

    :param i: Número do dataset, usado para nomear os gráficos.
    :param n: Número de iterações do treinamento
    :param history: Histórico de treinamento do modelo sem transfer learning (objeto History do Keras).
    :param history_transfer: Histórico de treinamento do modelo com transfer learning (objeto History do Keras).
    :param test: Resultados do teste do modelo sem transfer learning no dataset de treino.
    :param test_transfer: Resultados do teste do modelo com transfer learning no dataset de treino.
    :param test_others: Resultados do teste do modelo sem transfer learning nos datasets não utilizados no treino.
    :param test_transfer_others: Resultados do teste do modelo com transfer learning nos datasets não utilizados no treino.
    """
    dir = Path(f"./resultados/graphs/dataset{i}")

    if not dir.exists():
        dir.mkdir()

    # Plotar perda sem transfer learning
    grafico_aux(i=i, titulo=f"Perda Com o Dataset {i}", treino=history.history['loss'],
                validacao=history.history['val_loss'], teste=test, teste_other=test_others, ylabel="Perda",
                output_path=dir / f"perda{n}.svg")

    # Plotar perda com transfer learning
    grafico_aux(i=i, titulo=f"Perda Com o Dataset {i} Com Transfer Learning", treino=history_transfer.history['loss'],
                validacao=history_transfer.history['val_loss'], teste=test_transfer, teste_other=test_transfer_others,
                ylabel="Perda", output_path=dir / f"perda_transfer{n}.svg")

    # Plotar acurácia sem transfer learning
    grafico_aux(i=i, titulo=f"Acurácia Com o Dataset {i}", treino=history.history['accuracy'],
                validacao=history.history['val_accuracy'], teste=test, teste_other=test_others, ylabel="Acurácia",
                output_path=dir / f"acuracia{n}.svg")

    # Plotar acurácia com transfer learning
    grafico_aux(i=i, titulo=f"Acurácia Com o Dataset {i} Com Transfer Learning",
                treino=history_transfer.history['accuracy'], validacao=history_transfer.history['val_accuracy'],
                teste=test_transfer, teste_other=test_transfer_others, ylabel="Acurácia",
                output_path=dir / f"acuracia_transfer{n}.svg")


def salvar_resultados(metrics: list[str], N: list[list[int]], i: int, test_score: dict[str:float],
                      test_score_transfer: dict[str:float], time: timedelta, time_transfer: timedelta,
                      test_scores_others: dict[str:float], test_scores_transfer_others: dict[str:float]) -> None:
    """
    Salva os resultados do treinamento e teste em arquivos.

    :param metrics: Lista com os nomes das métricas a serem salvas
    :param N: Uma tupla contendo duas listas de inteiros, onde a primeira lista representa o número de exemplos
    positivos e a segunda lista o número de exemplos negativos para cada dataset.
    :param i: Número do dataset, usado para nomear o arquivo de resultados.
    :param test_score: Pontuação do teste do modelo sem transfer learning (uma tupla com perda na posição 0 e acurácia
    na posição 1).
    :param test_score_transfer: Pontuação do teste do modelo com transfer learning (uma tupla com perda na posição 0 e
    acurácia na posição 1).
    :param time: Tempo decorrido para o treinamento sem transfer learning.
    :param time_transfer: Tempo decorrido para o treinamento com transfer learning.
    :param test_scores_others: Dicionário de pontuações de testes com outros datasets (em que as chaves são os outros datasets).
    :param test_scores_transfer_others: Dicionário de pontuações de testes com outros datasets usando transfer learning (em que as chaves são os outros datasets).
    :return: None
    """
    resultados_path = Path("resultados")
    resultados_path.mkdir(exist_ok=True)

    info_path = resultados_path / f"info{i}.txt"
    tempo_path = resultados_path / f"tempo/tempo{i}.csv"
    testes_path = resultados_path / f"testes/testes{i}.csv"
    others_path = resultados_path / f"testes/others{i}.csv"

    if not info_path.exists():
        with info_path.open("w") as file:
            file.write(f"N: {N[i - 1][0] + N[i - 1][1]} imagens\n"
                       f"N positivas: {N[i - 1][0]} imagens\n"
                       f"N negativas: {N[i - 1][1]} imagens")

    with tempo_path.open("a") as file:
        file.write(f"{time},{time_transfer}\n")

    # função auxiliar para salvar as métricas em um arquivo
    def salvar_metricas(file_path: Path, *resultados):
        line = []
        for resultado in resultados:
            for metric in metrics:
                line.append(resultado[metric])
        with file_path.open("a") as file:
            file.write(";".join(str(m) for m in line) + "\n")

    salvar_metricas(testes_path, test_score, test_score_transfer)

    j = 3 - i
    ds = f"dataset{j}"
    salvar_metricas(others_path, test_scores_others[ds], test_scores_transfer_others[ds])
