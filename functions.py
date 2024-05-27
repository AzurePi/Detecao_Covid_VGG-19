import os
import shutil
from datetime import timedelta
from pathlib import Path

import keras.callbacks
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def salvar_parametros(base_learning_rate: float, epocas: int, test_split: float, validation_split: float) -> None:
    """
    Salva os parâmetros do treinamento em um arquivo "resultados/parametros.txt".

    :param base_learning_rate: Taxa de aprendizado utilizada no treinamento.
    :param epocas: Número de épocas do treinamento.
    :param test_split: Proporção dos dados utilizados para teste.
    :param validation_split: Proporção dos dados utilizados para validação.
    :return: None
    """
    with open("./resultados/parametros.txt", 'w') as file:
        file.write(f"Learning rate: {base_learning_rate}\n")
        file.write(f"Épocas: {epocas}\n")
        file.write(f"Test split: {test_split}\n")
        file.write(f"Validation split: {validation_split}\n")
        file.close()


def prepararResultados() -> tuple[Path, int]:
    """
    Prepara a estrutura de diretórios para armazenar os resultados do treinamento. Calcula o número de treinamentos
    já feitos com base na quantidade de arquivos em "./resultados/training/normal".

    Verifica se os diretórios "./resultados", "./resultados/training", "./resultados/training/normal" e
    "./resultados/training/transfer" existem. Se não existirem, cria esses diretórios.

    :return: Uma tupla contendo o caminho do diretório de treinamento e o número de treinamentos já feitos.
    """
    if not Path("./resultados").exists():
        Path("./resultados").mkdir()
    if not Path("./resultados/training").exists():
        Path("./resultados/training").mkdir()
    if not Path("./resultados/training/normal").exists():
        Path("./resultados/training/normal").mkdir()
    if not Path("./resultados/training/transfer").exists():
        Path("./resultados/training/transfer").mkdir()

    training = Path("./resultados/training")
    n = np.ceil(len(list((training / "normal").glob("*"))) / 2)
    return training, n


def prepararDiretorios(dataset_dir1: Path, dataset_dir2: Path, test_split: float) -> tuple[list[int], list[int]]:
    """
    Prepara os diretórios e conjuntos de dados para treinamento e teste.

    :param dataset_dir1: Diretório do primeiro conjunto de dados.
    :param dataset_dir2: Diretório do segundo conjunto de dados.
    :param test_split: Proporção dos dados a serem utilizados para teste.
    :return: Uma lista contendo duas sublistas, cada uma com o número de exemplos positivos e negativos para os dois conjuntos de dados.
    """

    preparar_dataset1(dataset_dir1)
    num_positivas1 = len(list((dataset_dir1 / "positivo").glob("*")))
    num_negativas1 = len(list((dataset_dir1 / "negativo").glob("*")))
    treinamento_e_teste(1, num_positivas1, num_negativas1, test_split, dataset_dir1)

    preparar_dataset2(dataset_dir2)
    num_positivas2 = len(list((dataset_dir2 / "positivo").glob("*")))
    num_negativas2 = len(list((dataset_dir2 / "negativo").glob("*")))
    treinamento_e_teste(2, num_positivas2, num_negativas2, test_split, dataset_dir2)

    return [num_positivas1, num_negativas1], [num_positivas2, num_negativas2]


def formatar_diretorio(origem: Path, destino: Path) -> None:
    """
    Move todos os arquivos de um diretório de origem para um destino e remove o diretório de origem.

    :param origem: Diretório de origem contendo os arquivos.
    :param destino: Diretório de destino.
    :return: None
    """
    if not destino.exists():
        destino.mkdir()
    for file in origem.iterdir():
        shutil.move(file, destino / file.name)
    shutil.rmtree(origem)


def preparar_dataset1(dataset_dir=Path("./dataset1")) -> None:
    """
    Prepara o ambiente com o dataset "sarscov2-ctscan-dataset", baixado do kaggle.

    Se o dataset ainda não foi baixado, baixamos e descompactamos.

    :param dataset_dir: Um diretório onde o dataset será armazenado.
    :return: None
    """
    if not dataset_dir.exists():
        print("Baixando dataset 1 de imagens e criando diretório...\n")
        dataset_dir.mkdir()

        os.system(f'kaggle datasets download -d plameneduardo/sarscov2-ctscan-dataset -p {dataset_dir} --unzip -q')

        positivo_dir = dataset_dir / "COVID"
        negativo_dir = dataset_dir / "non-COVID"

        formatar_diretorio(positivo_dir, dataset_dir / "positivo")
        formatar_diretorio(negativo_dir, dataset_dir / "negativo")

        print("Pronto!")
    else:
        print("Diretório de imagens para o dataset 1 já está presente na máquina. Prosseguindo...")


def preparar_dataset2(dataset_dir=Path("./dataset2")) -> None:
    """
    Prepara o ambiente com o dataset "preprocessed-ct-scans-for-covid19", baixado do kaggle.

    Se o dataset ainda não foi baixado, baixamos e descompactamos. Mantemos apenas as imagens originais,
    e não as pré-processadas.

    :param dataset_dir: Um diretório onde o dataset será armazenado.
    :return: None
    """
    if not dataset_dir.exists():
        print("Baixando dataset 2 de imagens e criando diretório...\n")
        dataset_dir.mkdir()

        os.system(f'kaggle datasets download -d azaemon/preprocessed-ct-scans-for-covid19 -p {dataset_dir} --unzip -q')
        shutil.rmtree(f"{dataset_dir}/Preprocessed CT scans")

        positivo_dir = dataset_dir / "Original CT Scans/pCT"
        negativo_dir = dataset_dir / "Original CT Scans/nCT"
        non_informative_dir = dataset_dir / "Original CT Scans/NiCT"

        formatar_diretorio(positivo_dir, dataset_dir / "positivo")
        formatar_diretorio(negativo_dir, dataset_dir / "negativo")
        formatar_diretorio(non_informative_dir, dataset_dir / "negativo")

        shutil.rmtree(dataset_dir / "Original CT Scans")
        print("Pronto!")
    else:
        print("Diretório de imagens pra o dataset 2 já está presente na máquina. Prosseguindo...")


def mover_imagens(origem: Path, num_imagens: int, destino: str) -> None:
    """
    Move um número especificado de imagens de um diretório de origem para um destino.

    :param origem: Diretório de origem contendo as imagens.
    :param num_imagens: O número de imagens para mover.
    :param destino: Nome do diretório de destino (incluindo subdiretório para a classe).
    :return: None
    """
    destino_path = Path(destino)
    if not destino_path.exists():
        destino_path.mkdir(parents=True)

    source_path = Path(origem)
    imagens = source_path.glob("*")

    for count, image_name in enumerate(imagens):
        if count >= num_imagens:
            break
        src = source_path / image_name.name
        dst = destino_path / image_name.name
        shutil.move(src, dst)


def treinamento_e_teste(n: int, num_positivas: int, num_negativas: int, test_split: float, dataset_dir: Path) -> None:
    """
    Verifica se os diretórios de treinamento e teste já existem. Se não existirem, cria um diretório temporário,
    copia as imagens do dataset original para esse diretório e então distribui as imagens entre os diretórios
    de treinamento e teste conforme a proporção "test_split".

    :param n: Número do dataset, utilizado para nomear os diretórios de treinamento e teste.
    :param num_positivas: Número de exemplos positivos no dataset original.
    :param num_negativas: Número de exemplos negativos no dataset original.
    :param test_split: Proporção dos dados a serem utilizados para teste.
    :param dataset_dir: Caminho do diretório contendo o dataset original.
    :return: None
    """
    if not (Path(f"treinamento{n}").exists() and Path(f"./teste{n}").exists()):
        print("\nCriando diretórios para treinamento e teste...")

        temp_dir = Path("./temp")
        if not temp_dir.exists():
            print("\tCriando diretório temporário...")
            temp_dir.mkdir()
            shutil.copytree(src=f"./{dataset_dir}/positivo", dst="./temp/positivo")
            shutil.copytree(src=f"./{dataset_dir}/negativo", dst="./temp/negativo")
            print("\tPronto!\nProsseguindo...")

        mover_imagens(temp_dir / "positivo", int(num_positivas * test_split), f"./teste{n}/positivo")
        mover_imagens(temp_dir / "negativo", int(num_negativas * test_split), f"./teste{n}/negativo")
        mover_imagens(temp_dir / "positivo", num_positivas - int(num_positivas * test_split),
                      f"./treinamento{n}/positivo")
        mover_imagens(temp_dir / "negativo", num_negativas - int(num_negativas * test_split),
                      f"./treinamento{n}/negativo")

        shutil.rmtree(temp_dir)
        print("Pronto!")
    else:
        print("Diretórios de treinamento e teste já estão presentes. Prosseguindo...")


def plotar_amostra(ds: tf.data.Dataset, filename: str) -> None:
    """
    Plota 9 imagens de um dataset e salva a figura.

    :param ds: Dataset de onde tirar as imagens.
    :param filename: Nome do arquivo para salvar a figura.
    """
    imagens, labels = next(iter(ds.take(1)))
    label_indices = np.argmax(labels, axis=1)
    rows, cols = 3, 3
    plt.figure(figsize=(cols * 2.5, rows * 2.5))
    for i in range(min(9, len(imagens))):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(imagens[i].numpy().astype("uint8"))
        plt.title(ds.class_names[label_indices[i]])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plotar_graficos(epocas: int, history: keras.callbacks.History, test_score: tuple[float, float],
                    history_transfer: keras.callbacks.History, test_score_transfer: tuple[float, float],
                    i: int, loss_max: float) -> None:
    """
    Plota os gráficos de acurácia e perda dos modelos treinados, com e sem transfer learning.
    Inclui pontos de acurácia e perda de teste.
    Garante que os eixos Y sejam os mesmos para todos os gráficos de acurácia e perda.

    :param epocas: Número de épocas do treinamento
    :param history: Histórico de treinamento do modelo sem transfer learning (objeto History do Keras).
    :param history_transfer: Histórico de treinamento do modelo com transfer learning (objeto History do Keras).
    :param test_score: Pontuação do teste do modelo sem transfer learning (vetor com perda na posição 0 e acurácia na posição 1).
    :param test_score_transfer: Pontuação do teste do modelo com transfer learning (vetor com perda na posição 0 e acurácia na posição 1).
    :param i: Número do dataset, usado para nomear os gráficos.
    :param loss_max: Máximo das perdas entre os treinamentos

    A função salva os seguintes gráficos:

    - Acurácia do modelo sem transfer learning: './resultados/acuracia{i}.png'
    - Perda do modelo sem transfer learning: './resultados/perda{i}.png'
    - Acurácia do modelo com transfer learning: './resultados/acuracia_transfer{i}.png'
    - Perda do modelo com transfer learning: './resultados/perda_transfer{i}.png'
    """
    loss_max += 0.1

    # plotar loss normal
    plt.title(f"Perda do Modelo {i}")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.scatter(epocas - 1, test_score[0], color='red')
    plt.annotate(f'{test_score[0]:.4f}', (epocas - 1, test_score[0]), textcoords="offset points", xytext=(0, 10),
                 ha='center', color='red')
    plt.xlabel("Época")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # garante que só inteiros serão utilizados na escala
    plt.ylabel("Perda")
    plt.ylim(0, loss_max)
    plt.legend(['train', 'validation', 'test'])
    plt.savefig(f"./resultados/perda{i}.png", bbox_inches='tight')
    plt.close()

    # plotar acurácia normal
    plt.title(f"Acurácia do Modelo {i}")
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.scatter(epocas - 1, test_score[1], color='red')
    plt.annotate(f'{test_score[1]:.4f}', (epocas - 1, test_score[1]), textcoords="offset points", xytext=(0, 10),
                 ha='center', color='red')
    plt.xlabel("Época")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # garante que só inteiros serão utilizados na escala
    plt.ylabel("Acurácia")
    plt.ylim(0, 1)
    plt.legend(['train', 'validation', 'test'])
    plt.savefig(f"./resultados/acuracia{i}.png", bbox_inches='tight')
    plt.close()

    # plotar loss com transfer learning
    plt.title(f"Perda do Modelo {i} Com Transfer Learning")
    plt.plot(history_transfer.history['loss'])
    plt.plot(history_transfer.history['val_loss'])
    plt.scatter(epocas - 1, test_score_transfer[0], color='red')
    plt.annotate(f'{test_score_transfer[0]:.4f}', (epocas - 1, test_score_transfer[0]), textcoords="offset points",
                 xytext=(0, 10), ha='center', color='red')
    plt.xlabel("Época")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # garante que só inteiros serão utilizados na escala
    plt.ylabel("Perda")
    plt.ylim(0, loss_max)
    plt.legend(['train', 'validation', 'test'])
    plt.savefig(f"./resultados/perda_transfer{i}.png", bbox_inches='tight')
    plt.close()

    # plotar acurácia com transfer learning
    plt.title(f"Acurácia do Modelo {i} Com Transfer Learning")
    plt.plot(history_transfer.history['accuracy'])
    plt.plot(history_transfer.history['val_accuracy'])
    plt.scatter(epocas - 1, test_score_transfer[1], color='red')
    plt.annotate(f'{test_score_transfer[1]:.4f}', (epocas - 1, test_score_transfer[1]), textcoords="offset points",
                 xytext=(0, 10), ha='center', color='red')
    plt.xlabel("Época")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # garante que só inteiros serão utilizados na escala
    plt.ylabel("Acurácia")
    plt.ylim(0, 1)
    plt.legend(['train', 'validation', 'test'])
    plt.savefig(f"./resultados/acuracia_transfer{i}.png", bbox_inches='tight')
    plt.close()


def salvar_resultados(N: tuple[list[int], list[int]], i: int, test_score: tuple[float, float],
                      test_score_transfer: tuple[float, float], time: timedelta,
                      time_transfer: timedelta) -> None:
    """
    Salva especificações do dataset e resultados do teste em um arquivo "resultados/info{i}.txt".

    :param N: Uma tupla contendo duas listas de inteiros, onde a primeira lista representa o número de exemplos
    positivos e a segunda lista o número de exemplos negativos para cada dataset.
    :param i: Número do dataset, usado para nomear o arquivo de resultados.
    :param test_score: Pontuação do teste do modelo sem transfer learning (uma tupla com perda na posição 0 e acurácia
    na posição 1).
    :param test_score_transfer: Pontuação do teste do modelo com transfer learning (uma tupla com perda na posição 0 e
    acurácia na posição 1).
    :param time: Tempo decorrido para o treinamento sem transfer learning (um objeto timedelta).
    :param time_transfer: Tempo decorrido para o treinamento com transfer learning (um objeto timedelta).
    :return: None
    """
    with open(f"resultados/info{i}.txt", "w") as file:
        file.write(f"N: {N[i - 1][0] + N[i - 1][1]} imagens\n")
        file.write(f"N positivas: {N[i - 1][0]} imagens\n")
        file.write(f"N negativas: {N[i - 1][1]} imagens\n\n")
        file.write(f"Tempo decorrido: {time}\n")
        file.write(f"Perda do teste: {test_score[0]}\n")
        file.write(f"Acurácia do teste: {test_score[1]}\n\n")
        file.write(f"Tempo decorrido com transfer learning: {time_transfer}\n")
        file.write(f"Perda do teste com transfer learning: {test_score_transfer[0]}\n")
        file.write(f"Acurácia do teste com transfer learning: {test_score_transfer[1]}\n")
        file.close()
