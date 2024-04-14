import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


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


def preparar_dataset(dataset_dir=Path("./dataset")):
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
    imagens = source_path.glob("*.jpg")

    for count, image_name in enumerate(imagens):
        if count >= num_imagens:
            break
        src = source_path / image_name.name
        dst = destino_path / image_name.name
        shutil.move(src, dst)


def treinamento_e_teste(num_positivas, num_negativas, test_split, dataset_dir=Path("./dataset")):
    if not (Path("treinamento").exists() & Path("./teste").exists()):
        print("\nCriando diretórios para treinamento e teste...")

        if not Path("./temp").exists():
            print("\tCriando diretório temporário...")
            Path("./temp").mkdir()
            with Pool() as pool:
                pool.apply_async(shutil.copytree, (f"./{dataset_dir}/positivo", "./temp/positivo"))
                pool.apply_async(shutil.copytree, (f"./{dataset_dir}/negativo", "./temp/negativo"))
                pool.close()
                pool.join()
            print("\tPronto!\nProsseguindo...")

        mover_imagens("./temp/positivo", int(num_positivas * test_split), "./teste/positivo")
        mover_imagens("./temp/negativo", int(num_negativas * test_split), "./teste/negativo")
        mover_imagens("./temp/positivo", num_positivas - int(num_positivas * test_split), "./treinamento/positivo")
        mover_imagens("./temp/negativo", num_negativas - int(num_negativas * test_split), "./treinamento/negativo")

        shutil.rmtree("./temp")
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
