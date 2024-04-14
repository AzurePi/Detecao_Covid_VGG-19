# Importações ----------------------------------------------------------------------------------------------------------
from datetime import datetime
from pathlib import Path
from random import seed, randint

import tensorflow as tf
from keras.callbacks import CSVLogger
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from functions import plot_images_from_dataset, preparar_dataset, treinamento_e_teste

seed()

# Parâmetros do treinamento --------------------------------------------------------------------------------------------
test_split = 0.2
validation_split = 0.4

base_learning_rate = 0.001
epocas = 100

# Preparação dos diretórios --------------------------------------------------------------------------------------------
dataset_dir = Path("./dataset")
preparar_dataset(dataset_dir)

if not Path("resultados").exists():
    Path("resultados").mkdir()

num_positivas = len(list((dataset_dir / "positivo").glob("*.jpg")))
num_negativas = len(list((dataset_dir / "negativo").glob("*.jpg")))

treinamento_e_teste(num_positivas, num_negativas, test_split, dataset_dir)

# Criação de datasets keras --------------------------------------------------------------------------------------------
print("\nCriando datasets a partir dos diretórios...")

train_ds, validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory="./treinamento",
    labels="inferred",
    label_mode="categorical",
    seed=randint(0, 100),
    subset="both",
    validation_split=validation_split,
    batch_size=32,
    image_size=(256, 256))

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory="./teste",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(256, 256))

print("Pronto!\n")

# Aplicação da VGG-19 --------------------------------------------------------------------------------------------------

# criar a base do modelo
image_inputs = tf.keras.layers.Input(shape=(256, 256, 3))
VGG_19_base = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_tensor=image_inputs,
                                          pooling="max", classifier_activation="softmax")

# congelar a base para transfer learning
for i, layer in enumerate(VGG_19_base.layers):
    layer.trainable = False

# layers para fine-tuning
FC_layer_Flatten = tf.keras.layers.Flatten()(VGG_19_base.output)
x = tf.keras.layers.Dropout(0.3)(FC_layer_Flatten)
x = tf.keras.layers.Dense(units=128, activation="relu")(x)
x = tf.keras.layers.Dense(units=32, activation="relu")(x)
x = tf.keras.layers.Dense(units=8, activation="relu")(x)
outputs = tf.keras.layers.Dense(units=2, activation="softmax")(x)

# compilar o modelo final
modelo_final = tf.keras.Model(inputs=image_inputs, outputs=outputs)
modelo_final.summary()
print("\n")

modelo_final.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                     loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

# criar um log em CSV para armazenar a história do modelo
csv_logger = CSVLogger("resultados/training.csv", ",", False)

# treinamento
print("--------------------------- Treinamento do modelo ---------------------------")
start_time = datetime.now()
history = modelo_final.fit(train_ds, validation_data=validation_ds, epochs=epocas, callbacks=[csv_logger], verbose=2)
end_time = datetime.now()

# teste
print("\n------------------------------ Teste do modelo ------------------------------")
test_score = modelo_final.evaluate(test_ds)

# Registro dos resultados ----------------------------------------------------------------------------------------------
print("Salvando informações e resultados...")

# plotar uma imagem com exemplos de imagesn positivas e negativas
plot_images_from_dataset(test_ds, "resultados/sample.png")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Acurácia do Modelo")
plt.xlabel("Épocas")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # garante que só inteiros serão utilizados na escala
plt.ylabel("Acurácia")
plt.legend(['train', 'validation'])
plt.savefig("./resultados/acuracia.png", bbox_inches='tight')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Perda do Modelo")
plt.xlabel("Época")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # garante que só inteiros serão utilizados na escala
plt.ylabel("Perda")
plt.legend(['train', 'validation'])
plt.savefig("./resultados/perda.png", bbox_inches='tight')
plt.close()

with open("resultados/info.txt", "w") as file:
    file.write(f"N: {num_positivas + num_negativas} imagens\n")
    file.write(f"Test split: {test_split}\n")
    file.write(f"Validation split: {validation_split}\n")
    file.write(f"Learning rate: {base_learning_rate}\n")
    file.write(f"Épocas: {epocas}\n")
    file.write(f"Tempo decorrido: {end_time - start_time}\n")
    file.write(f"Acurácia do deste: {test_score[1]}\n")
    file.write(f"Perda do deste: {test_score[0]}")

print("Pronto! Cheque a pasta resultados.")
