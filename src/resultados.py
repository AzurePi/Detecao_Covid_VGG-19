from datetime import timedelta
from pathlib import Path

import keras.callbacks
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


def prepararResultados() -> tuple[Path, int]:
    """
    Prepara a estrutura de diretórios para armazenar os resultados do treinamento. Calcula o número de treinamentos
    já feitos com base na quantidade de arquivos em "./resultados/logs/normal".

    Verifica se os diretórios "./resultados", "./resultados/logs", "./resultados/logs/normal" e
    "./resultados/logs/transfer" existem. Se não existirem, cria esses diretórios.

    :return: Uma tupla contendo o caminho do diretório de treinamento e o número de treinamentos já feitos (calculado
    com base no número de logs encontrados)
    """
    dirs = ["./resultados", "./resultados/logs", "./resultados/logs/normal", "./resultados/logs/transfer",
            "./resultados/graphs", "./resultados/tempo", "./resultados/testes"]

    print("Preparando estrutura do diretório de resultados...\n")

    for path in dirs:
        Path(path).mkdir(parents=True, exist_ok=True)

    for i in [1, 2, 3]:
        csv_files = [
            (f"./resultados/tempo/tempo{i}.csv", "Tempo,Tempo Transfer\n"),
            (f"./resultados/testes/testes{i}.csv", "Perda,Acurácia,Perda Transfer,Acurácia Transfer\n"),
            (f"./resultados/testes/others{i}.csv", "".join(
                f"Perda dataset{j},Acurácia dataset{j},Perda Transfer dataset{j},Acurácia Transfer dataset{j}"
                for j in [1, 2, 3] if j != i) + "\n")
        ]

        for path, header in csv_files:
            if not Path(path).exists():
                with open(path, "w") as file:
                    file.write(header)

    training = Path("./resultados/logs/normal")
    n = sum(1 for _ in training.iterdir()) // 2
    return Path("./resultados/logs"), n


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


def grafico_aux(i: int, titulo: str, treino, validacao, teste, teste_other, ylabel: str, ylim: tuple, output_path: Path,
                epoca_excede=None) -> None:
    """
    Função auxiliar para plotar gráficos de perda e acurácia.

    :param i: Número do dataset.
    :param titulo: Título do gráfico.
    :param treino: Dados de treino.
    :param validacao: Dados de validação.
    :param teste: Dados de teste.
    :param teste_other: Dados de teste com os demais datasets.
    :param ylabel: Rótulo do eixo y.
    :param ylim: Limites superior do eixo y.
    :param output_path: Caminho para salvar a imagem.
    :param epoca_excede: Época em que a acurácia (se houver) excedeu 0.9.
    :return: None
    """
    m = 0.05
    v = 1

    if ylabel == "Perda":
        m = 0.25
        v = 0

    plt.title(titulo)
    plt.plot(treino, label="Treino")
    plt.plot(validacao, label="Validação")
    plt.xlabel("Época")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # garante que só inteiros serão utilizados na escala
    plt.ylabel(ylabel)
    plt.ylim(*ylim)

    if epoca_excede is not None:
        plt.axvline(x=epoca_excede, color='purple', linestyle='--', label=f"Acima de 0.9 na época {epoca_excede}")

    plt.gca().yaxis.set_minor_locator(MultipleLocator(m))
    plt.plot(len(treino) - 1, teste[v], 'o', label=f"Teste (dataset {i})")

    for j in [x for x in [1, 2, 3] if x != i]:
        plt.plot(len(treino) - 1, teste_other[f"dataset{j}"][v], 'o', label=f"Teste (dataset {j})")

    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.legend(shadow=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plotar_graficos(i: int, n: int, loss_max: float, history: keras.callbacks.History,
                    history_transfer: keras.callbacks.History,
                    epoca_acuracia_excede: int, epoca_acuracia_excede_transfer: int,
                    test, test_transfer, test_others, test_transfer_others) -> None:
    """
    Plota os gráficos de acurácia e perda dos modelos treinados, com e sem transfer learning. Separa os gráficos em
    pastas diferentes segundo o dataset utilizado.


    Garante que os eixos Y sejam os mesmos para todos os gráficos de acurácia e perda.

    :param i: Número do dataset, usado para nomear os gráficos.
    :param n: Número de iterações do treinamento
    :param loss_max: Máximo das perdas entre os treinamentos
    :param history: Histórico de treinamento do modelo sem transfer learning (objeto History do Keras).
    :param history_transfer: Histórico de treinamento do modelo com transfer learning (objeto History do Keras).
    :param epoca_acuracia_excede: Época em que a acurácia de validação excedeu 0.9.
    :param epoca_acuracia_excede_transfer: Época em que a acurácia de validação com transfer learning excedeu 0.9.
    :param test: Resultados do teste do modelo sem transfer learning no dataset de treino.
    :param test_transfer: Resultados do teste do modelo com transfer learning no dataset de treino.
    :param test_others: Resultados do teste do modelo sem transfer learning nos datasets não utilizados no treino.
    :param test_transfer_others: Resultados do teste do modelo com transfer learning nos datasets não utilizados no treino.
    """
    loss_max += 0.1
    dir = Path(f"./resultados/graphs/dataset{i}")

    if not dir.exists():
        dir.mkdir()

    # Plotar perda sem transfer learning
    grafico_aux(i=i,
                titulo=f"Perda Com o Dataset {i}",
                treino=history.history['loss'], validacao=history.history['val_loss'],
                teste=test, teste_other=test_others,
                ylabel="Perda", ylim=(0, loss_max),
                output_path=dir / f"perda ({n}).png")

    # Plotar perda com transfer learning
    grafico_aux(i=i,
                titulo=f"Perda Com o Dataset {i} Com Transfer Learning",
                treino=history_transfer.history['loss'], validacao=history_transfer.history['val_loss'],
                teste=test_transfer, teste_other=test_transfer_others,
                ylabel="Perda", ylim=(0, loss_max),
                output_path=dir / f"perda_transfer ({n}).png")

    # Plotar acurácia sem transfer learning
    grafico_aux(i=i,
                titulo=f"Acurácia Com o Dataset {i}",
                treino=history.history['accuracy'], validacao=history.history['val_accuracy'],
                epoca_excede=epoca_acuracia_excede,
                teste=test, teste_other=test_others,
                ylabel="Acurácia", ylim=(0, 1.05),
                output_path=dir / f"acuracia ({n}).png")

    # Plotar acurácia com transfer learning
    grafico_aux(i=i,
                titulo=f"Acurácia Com o Dataset {i} Com Transfer Learning",
                treino=history_transfer.history['accuracy'], validacao=history_transfer.history['val_accuracy'],
                epoca_excede=epoca_acuracia_excede_transfer,
                teste=test_transfer, teste_other=test_transfer_others,
                ylabel="Acurácia", ylim=(0, 1.05),
                output_path=dir / f"acuracia_transfer ({n}).png")


def salvar_resultados(N: list[list[int]], i: int, test_score: tuple[float, float],
                      test_score_transfer: tuple[float, float], time: timedelta, time_transfer: timedelta,
                      test_scores_others: dict, test_scores_transfer_others: dict, exc, transfer) -> None:
    """
    Salva especificações do dataset em um arquivo "resultados/info{i}.txt".

    Salva o tempo decorrido no teste no arquivo "resultados/tempo{i}.csv".

    Salva os valores de acurácia e perda do teste em um arquivo "resultados/teste{i}.csv".

    :param exc:
    :param transfer:
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
    if not Path(f"resultados/info{i}.txt").exists():
        with open(f"resultados/info{i}.txt", "w") as file:
            file.write(f"N: {N[i - 1][0] + N[i - 1][1]} imagens\n"
                       f"N positivas: {N[i - 1][0]} imagens\n"
                       f"N negativas: {N[i - 1][1]} imagens")

    with open(f"resultados/tempo/tempo{i}.csv", "a") as file:
        file.write(f"{time},{time_transfer}\n")

    with open(f"resultados/testes/testes{i}.csv", "a") as file:
        file.write(f"{test_score[0]},{test_score[1]},{test_score_transfer[0]},{test_score_transfer[1]}\n")

    with open(f"resultados/testes/others{i}.csv", "a") as file:
        for j in [x for x in [1, 2, 3] if x != i]:
            ds = f"dataset{j}"
            file.write(
                f"{test_scores_others[ds][0]},"
                f"{test_scores_others[ds][1]},"
                f"{test_scores_transfer_others[ds][0]},"
                f"{test_scores_transfer_others[ds][1]}"
                "\n")
