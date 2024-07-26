import shutil
from pathlib import Path
from random import seed, randint

import kaggle
import tensorflow as tf


def carregar_datasets(i: int, validation_split: float, batch_size: int) -> tuple[
    tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """
    Carrega e prepara datasets de treinamento, validação e teste.

    :param i: Número do dataset.
    :param validation_split: Proporção dos dados de treinamento a serem utilizados para validação.
    :param batch_size: Tamanho da batch para o carregamento dos dados.
    :return: Tupla contendo os datasets de treinamento, validação, teste e uma lista dos nomes das classes.
    """
    seed()
    random = randint(0, 100)

    labels, label_mode = "inferred", "categorical"
    image_size = (256, 256)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=f"./treinamento{i}",
        labels=labels,
        label_mode=label_mode,
        subset="training",
        seed=random,
        validation_split=validation_split,
        batch_size=batch_size,
        image_size=image_size
    )

    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=f"./treinamento{i}",
        labels=labels,
        label_mode=label_mode,
        subset="validation",
        seed=random,
        validation_split=validation_split,
        batch_size=batch_size,
        image_size=image_size
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=f"./teste{i}",
        labels=labels,
        label_mode=label_mode,
        batch_size=batch_size,
        image_size=image_size
    )

    class_names = train_ds.class_names

    # Otimização com cache e prefetch
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, validation_ds, test_ds, class_names


def preparar_diretorios(test_split: float, dataset_dirs=[Path("./dataset1"), Path("./dataset2"), Path("./dataset3")]) -> \
        list[list[int]]:
    """
    Prepara os diretórios e conjuntos de dados para treinamento e teste.

    Verifica se os diretórios de datasets específicos existem. Se não, prepara os datasets necessários e
    cria os diretórios de treinamento e teste dividindo as imagens conforme a proporção especificada para teste.

    :param test_split: Proporção dos dados a serem utilizados para teste.
    :param dataset_dirs: Lista dos caminhos para os diretórios de imagens.
    :return: Uma lista contendo sublistas com o número de exemplos positivos e negativos para cada dataset.
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
    Prepara o ambiente com o dataset "`sarscov2-ctscan-dataset`", baixado do kaggle.

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
    Prepara o ambiente com o dataset "`preprocessed-ct-scans-for-covid19`", baixado do kaggle.

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


def preparar_dataset3(dataset_dir1=Path("./dataset1"), dataset_dir2=Path("./dataset2"),
                      dataset_dir3=Path("./dataset3")) -> None:
    """
    Prepara o ambiente com um dataset que une imagens do dataset1 e do dataset2.

    :param dataset_dir1: Diretório do dataset 1
    :param dataset_dir2: Diretório do dataset 2
    :param dataset_dir3: Um diretório onde o dataset unido será armazenado.
    :return: None
    """
    if not dataset_dir3.exists():
        print("Unindo diretórios 1 e 2 e criando diretório 3...")

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


def apagar_treinamento_e_teste():
    pai = Path("./")
    deletar = []

    for diretorio in pai.glob("*"):
        if diretorio.is_dir() and diretorio.name.startswith("teste") or diretorio.name.startswith("treinamento"):
            deletar.append(diretorio)

    for diretorio in deletar:
        shutil.rmtree(diretorio)
