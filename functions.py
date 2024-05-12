import os
import shutil
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def prepararDiretorios(dataset_dir1, dataset_dir2, test_split):
    if not Path("resultados").exists():
        Path("resultados").mkdir()

    preparar_dataset1(dataset_dir1)
    num_positivas1 = len(list((dataset_dir1 / "positivo").glob("*")))
    num_negativas1 = len(list((dataset_dir1 / "negativo").glob("*")))
    treinamento_e_teste(1, num_positivas1, num_negativas1, test_split, dataset_dir1)

    preparar_dataset2(dataset_dir2)
    num_positivas2 = len(list((dataset_dir2 / "positivo").glob("*")))
    num_negativas2 = len(list((dataset_dir2 / "negativo").glob("*")))
    treinamento_e_teste(2, num_positivas2, num_negativas2, test_split, dataset_dir2)

    return num_positivas1 + num_negativas1, num_positivas2 + num_negativas2


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


def preparar_dataset1(dataset_dir):
    """
    Prepara o ambiente com o dataset "sarscov2-ctscan-dataset", baixado do kaggle

    Se o dataset ainda não foi baixado, baixamos e descompactamos

    :param dataset_dir: Um diretório pathlib.Path onde o dataset sera armazenado
    :return:
    """
    if not dataset_dir.exists():
        print("Baixando dataset de imagens e criando diretório...\n")
        dataset_dir.mkdir()

        os.system(f'kaggle datasets download -d plameneduardo/sarscov2-ctscan-dataset -p {dataset_dir} --unzip -q')

        positivo_dir = dataset_dir / "COVID"
        negativo_dir = dataset_dir / "non-COVID"

        formatar_diretorio(positivo_dir, dataset_dir / "positivo")
        formatar_diretorio(negativo_dir, dataset_dir / "negativo")

        print("Pronto!")
    else:
        print("Diretório de imagens já está presente na máquina. Prosseguindo...")


def preparar_dataset2(dataset_dir=Path("./dataset2")):
    """
    Prepara o ambiente com o dataset "preprocessed-ct-scans-for-covid19", baixado do kaggle

    Se o dataset ainda não foi baixado, baixamos e descompactamos

    Mantemos apenas as imagens originais, e não as pré-processadas

    :param dataset_dir: Um diretório pathlib.Path onde o dataset sera armazenado
    :return:
    """
    if not dataset_dir.exists():
        print("Baixando dataset de imagens e criando diretório...\n")
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
        print("Diretório de imagens já está presente na máquina. Prosseguindo...")


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

    source_path = Path(origem)
    imagens = source_path.glob("*")

    for count, image_name in enumerate(imagens):
        if count >= num_imagens:
            break
        src = source_path / image_name.name
        dst = destino_path / image_name.name
        shutil.move(src, dst)


# padrao_regex = re.compile(r".*\.(jpg|jpeg|png)$", re.IGNORECASE)
# if padrao_regex.match(image_name.name):


def treinamento_e_teste(n, num_positivas, num_negativas, test_split, dataset_dir):
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
        mover_imagens(temp_dir / "/negativo", int(num_negativas * test_split), f"./teste{n}/negativo")
        mover_imagens(temp_dir / "/positivo", num_positivas - int(num_positivas * test_split),
                      f"./treinamento{n}/positivo")
        mover_imagens(temp_dir / "/negativo", num_negativas - int(num_negativas * test_split),
                      f"./treinamento{n}/negativo")

        shutil.rmtree(temp_dir)
        print("Pronto!")
    else:
        print("Diretórios de treinamento e teste já estão presentes. Prosseguindo...")


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
        plt.subplot(rows, cols, i + 1)
        plt.imshow(imagens[i].numpy().astype("uint8"))
        plt.title(ds.class_names[label_indices[i]])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
