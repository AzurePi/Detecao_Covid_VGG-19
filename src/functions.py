import shutil
from datetime import timedelta
from pathlib import Path
from typing import Tuple, List

import kaggle
import keras.callbacks
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


def prepararResultados() -> tuple[Path, int]:
    """
    Prepara a estrutura de diretórios para armazenar os resultados do treinamento. Calcula o número de treinamentos
    já feitos com base na quantidade de arquivos em "./resultados/logs/normal".

    Verifica se os diretórios "./resultados", "./resultados/logs", "./resultados/logs/normal" e
    "./resultados/logs/transfer" existem. Se não existirem, cria esses diretórios.

    :return: Uma tupla contendo o caminho do diretório de treinamento e o número de treinamentos já feitos 9calculado com base no número de logs encontrados)
    """
    dirs = ["./resultados", "./resultados/logs", "./resultados/logs/normal", "./resultados/logs/transfer",
            "./resultados/graphs", "./resultados/tempo", "./resultados/testes"]

    for path in dirs:
        Path(path).mkdir(parents=True, exist_ok=True)

    for i in [1, 2, 3]:
        if not Path(f"./resultados/tempo/tempo{i}.csv").exists():
            with open(f"./resultados/tempo/tempo{i}.csv", "w") as file:
                file.write("Tempo,Tempo Transfer\n")

        if not Path(f"./resultados/testes/testes{i}.csv").exists():
            with open(f"./resultados/testes/testes{i}.csv", "w") as file:
                file.write("Perda,Acurácia,Perda Transfer,Acurácia Transfer\n")

    training = Path("./resultados/logs/normal")
    n = sum(1 for _ in training.iterdir()) // 2
    return Path("./resultados/logs"), n


def prepararDiretorios(test_split: float, dataset_dirs=[Path("dataset1"), Path("dataset2"), Path("dataset3")]) -> list[list[int]]:
    """
    Prepara os diretórios e conjuntos de dados para treinamento e teste.

    :param test_split: Proporção dos dados a serem utilizados para teste.
    :param dataset_dirs: Lista dos caminhos para os diretórios de imagens.
    :return: Uma lista contendo três sublistas, cada uma com o número de exemplos positivos e negativos para os conjuntos de dados.
    """
    N = []

    for i, dataset_dir in enumerate(dataset_dirs, start=1):
        if i == 1:
            preparar_dataset1(dataset_dir)
        elif i == 2:
            preparar_dataset2(dataset_dir)
        elif i == 3:
            preparar_dataset3(*dataset_dirs)

        num_positivas = len(list((dataset_dir / "positivo").glob("*")))
        num_negativas = len(list((dataset_dir / "negativo").glob("*")))
        treinamento_e_teste(i, num_positivas, num_negativas, test_split, dataset_dir)
        N.append([num_positivas, num_negativas])

    return N


def formatar_diretorio(origem: Path, destino: Path) -> None:
    """
    Move todos os arquivos de um diretório de origem para um destino e remove o diretório de origem.

    :param origem: Diretório de origem contendo os arquivos.
    :param destino: Diretório de destino.
    :return: None
    """
    destino.mkdir(parents=True, exist_ok=True)
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
        print("Baixando dataset 1 de imagens e criando diretório...")
        dataset_dir.mkdir()

        kaggle.api.dataset_download_files('plameneduardo/sarscov2-ctscan-dataset', path=dataset_dir, unzip=True)

        positivo_dir = dataset_dir / "COVID"
        negativo_dir = dataset_dir / "non-COVID"

        formatar_diretorio(positivo_dir, dataset_dir / "positivo")
        formatar_diretorio(negativo_dir, dataset_dir / "negativo")

        print("Pronto!\n")
    else:
        print("Diretório de imagens para o dataset 1 já está presente na máquina. Prosseguindo...\n")


def preparar_dataset2(dataset_dir=Path("./dataset2")) -> None:
    """
    Prepara o ambiente com o dataset "preprocessed-ct-scans-for-covid19", baixado do kaggle.

    Se o dataset ainda não foi baixado, baixamos e descompactamos. Mantemos apenas as imagens originais,
    e não as pré-processadas.

    :param dataset_dir: Um diretório onde o dataset será armazenado.
    :return: None
    """
    if not dataset_dir.exists():
        print("Baixando dataset 2 de imagens e criando diretório...")
        dataset_dir.mkdir()

        kaggle.api.dataset_download_files('azaemon/preprocessed-ct-scans-for-covid19', path=dataset_dir, unzip=True)
        shutil.rmtree(dataset_dir / "Preprocessed CT scans")

        positivo_dir = dataset_dir / "Original CT Scans/pCT"
        negativo_dir = dataset_dir / "Original CT Scans/nCT"
        non_informative_dir = dataset_dir / "Original CT Scans/NiCT"

        formatar_diretorio(positivo_dir, dataset_dir / "positivo")
        formatar_diretorio(negativo_dir, dataset_dir / "negativo")
        formatar_diretorio(non_informative_dir, dataset_dir / "negativo")

        shutil.rmtree(dataset_dir / "Original CT Scans")
        print("Pronto!\n")
    else:
        print("Diretório de imagens para o dataset 2 já está presente na máquina. Prosseguindo...\n")


def preparar_dataset3(dataset_dir1=Path("dataset1"), dataset_dir2=Path("dataset2"),
                      dataset_dir3=Path("dataset3")) -> None:
    """
    Prepara o ambiente com um dataset que une imagens do dataset1 e do dataset2.

    :param dataset_dir1: Diretório do dataset 1
    :param dataset_dir2: Diretório do dataset 2
    :param dataset_dir3: Um diretório onde o dataset unido será armazenado.
    :return: None
    """
    if not dataset_dir3.exists():
        print("Unindo diretórios 1 e 2 e criando diretório...")

        dataset_dir3.mkdir()
        (dataset_dir3 / "positivo").mkdir()
        (dataset_dir3 / "negativo").mkdir()

        for dataset in dataset_dir1, dataset_dir2:
            shutil.copytree(dataset / "positivo", dataset_dir3 / "positivo", dirs_exist_ok=True)
            shutil.copytree(dataset / "negativo", dataset_dir3 / "negativo", dirs_exist_ok=True)

        print("Pronto!\n")
    else:
        print("Diretório 3 unindo datasets 1 e 2 já está presente na máquina. Prosseguindo...\n")


def mover_imagens(origem: Path, num_imagens: int, destino: str) -> None:
    """
    Move um número especificado de imagens de um diretório de origem para um destino.

    :param origem: Diretório de origem contendo as imagens.
    :param num_imagens: O número de imagens para mover.
    :param destino: Nome do diretório de destino (incluindo subdiretório para a classe).
    :return: None
    """
    destino_path = Path(destino)
    destino_path.mkdir(parents=True, exist_ok=True)

    for count, image_name in enumerate(origem.iterdir()):
        if count >= num_imagens:
            break
        shutil.move(origem / image_name.name, destino_path / image_name.name)


def treinamento_e_teste(i: int, num_positivas: int, num_negativas: int, test_split: float, dataset_dir: Path) -> None:
    """
    Verifica se os diretórios de treinamento e teste já existem. Se não existirem, cria um diretório temporário,
    copia as imagens do dataset original para esse diretório e então distribui as imagens entre os diretórios
    de treinamento e teste conforme a proporção "test_split".

    :param i: Número do dataset, utilizado para nomear os diretórios de treinamento e teste.
    :param num_positivas: Número de exemplos positivos no dataset original.
    :param num_negativas: Número de exemplos negativos no dataset original.
    :param test_split: Proporção dos dados a serem utilizados para teste.
    :param dataset_dir: Caminho do diretório contendo o dataset original.
    :return: None
    """
    treino_dir = Path(f"treinamento{i}")
    teste_dir = Path(f"teste{i}")

    if not (treino_dir.exists() and teste_dir.exists()):
        print(f"Criando diretórios para treinamento e teste {i}...")

        temp_dir = Path("./temp")
        temp_dir.mkdir(exist_ok=True)

        if not (temp_dir / "positivo").exists():
            print("\tCriando diretório temporário...")
            shutil.copytree(dataset_dir / "positivo", temp_dir / "positivo")
            shutil.copytree(dataset_dir / "negativo", temp_dir / "negativo")
            print("\tPronto!\nProsseguindo...")

        mover_imagens(temp_dir / "positivo", int(num_positivas * test_split), (teste_dir / "positivo").__str__())
        mover_imagens(temp_dir / "negativo", int(num_negativas * test_split), (teste_dir / "negativo").__str__())
        mover_imagens(temp_dir / "positivo", num_positivas - int(num_positivas * test_split),
                      (treino_dir / "positivo").__str__())
        mover_imagens(temp_dir / "negativo", num_negativas - int(num_negativas * test_split),
                      (treino_dir / "negativo").__str__())

        shutil.rmtree(temp_dir)
        print("Pronto!\n")
    else:
        print(f"Diretórios de treinamento e teste {i} já estão presentes. Prosseguindo...\n")


def plotar_amostra(ds: tf.data.Dataset, filename: str, class_names: list[str]) -> None:
    """
    Plota 9 imagens de um dataset e salva a figura.

    :param ds: Dataset de onde tirar as imagens.
    :param filename: Nome do arquivo para salvar a figura.
    :param class_names: Nomes das classes às quais as imagens podem pertencer
    """
    rows, cols = 3, 3
    plt.figure(figsize=(cols * 2.5, rows * 2.5))
    for images, labels in ds.take(1):
        for i in range(min(9, len(images))):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i].numpy().argmax()])
            plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plotar_graficos(n: int, history: keras.callbacks.History, history_transfer: keras.callbacks.History, i: int,
                    loss_max: float) -> None:
    """
    Plota os gráficos de acurácia e perda dos modelos treinados, com e sem transfer learning.
    Inclui pontos de acurácia e perda de teste.
    Garante que os eixos Y sejam os mesmos para todos os gráficos de acurácia e perda.

    :param n: Número de iterações do treinamento
    :param history: Histórico de treinamento do modelo sem transfer learning (objeto History do Keras).
    :param history_transfer: Histórico de treinamento do modelo com transfer learning (objeto History do Keras).
    :param i: Número do dataset, usado para nomear os gráficos.
    :param loss_max: Máximo das perdas entre os treinamentos

    A função salva os seguintes gráficos:

    - Acurácia do modelo sem transfer learning: './resultados/acuracia{i}.png'
    - Perda do modelo sem transfer learning: './resultados/perda{i}.png'
    - Acurácia do modelo com transfer learning: './resultados/acuracia_transfer{i}.png'
    - Perda do modelo com transfer learning: './resultados/perda_transfer{i}.png'
    """
    loss_max += 0.1
    dir = Path(f"./resultados/graphs/dataset{i}")

    if not dir.exists():
        dir.mkdir()

    # plotar loss normal
    plt.title(f"Perda do Modelo {i}")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel("Época")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # garante que só inteiros serão utilizados na escala
    plt.ylabel("Perda")
    plt.ylim(0, loss_max)
    plt.legend(['train', 'validation', 'test'])
    plt.savefig(dir / f"perda ({n}).png", bbox_inches='tight')
    plt.close()

    # plotar acurácia normal
    plt.title(f"Acurácia do Modelo {i}")
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel("Época")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # garante que só inteiros serão utilizados na escala
    plt.ylabel("Acurácia")
    plt.ylim(0, 1)
    plt.legend(['train', 'validation', 'test'])
    plt.savefig(dir / f"acuracia ({n}).png", bbox_inches='tight')
    plt.close()

    # plotar loss com transfer learning
    plt.title(f"Perda do Modelo {i} Com Transfer Learning")
    plt.plot(history_transfer.history['loss'])
    plt.plot(history_transfer.history['val_loss'])
    plt.xlabel("Época")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # garante que só inteiros serão utilizados na escala
    plt.ylabel("Perda")
    plt.ylim(0, loss_max)
    plt.legend(['train', 'validation', 'test'])
    plt.savefig(dir / f"perda_transfer ({n}).png", bbox_inches='tight')
    plt.close()

    # plotar acurácia com transfer learning
    plt.title(f"Acurácia do Modelo {i} Com Transfer Learning")
    plt.plot(history_transfer.history['accuracy'])
    plt.plot(history_transfer.history['val_accuracy'])
    plt.xlabel("Época")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # garante que só inteiros serão utilizados na escala
    plt.ylabel("Acurácia")
    plt.ylim(0, 1)
    plt.legend(['train', 'validation', 'test'])
    plt.savefig(dir / f"acuracia_transfer ({n}).png", bbox_inches='tight')
    plt.close()


def salvar_resultados(N: list[list[int]], i: int, test_score: tuple[float, float],
                      test_score_transfer: tuple[float, float], time: timedelta,
                      time_transfer: timedelta) -> None:
    """
    Salva especificações do dataset em um arquivo "resultados/info{i}.txt".

    Salva o tempo decorrido no teste no arquivo "resutlados/tempo{i}.csv".

    Salva os valores de acurácia e perda do teste em um arquivo "resultados/teste{i}.csv".


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
    if not Path(f"resultados/info{i}.txt").exists():
        with open(f"resultados/info{i}.txt", "w") as file:
            file.write(f"N: {N[i - 1][0] + N[i - 1][1]} imagens\n")
            file.write(f"N positivas: {N[i - 1][0]} imagens\n")
            file.write(f"N negativas: {N[i - 1][1]} imagens")

    with open(f"resultados/tempo/tempo{i}.csv", "a") as file:
        file.write(f"{time},{time_transfer}\n")

    with open(f"resultados/testes/testes{i}.csv", "a") as file:
        file.write(f"{test_score[0]},{test_score[1]},{test_score_transfer[0]},{test_score_transfer[1]}\n")
