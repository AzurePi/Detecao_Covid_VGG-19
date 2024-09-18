Esse repositório contém o código fonte utilizado no projeto <b>Diagnóstico Automático Não-Invasivo de Covid-19 a partir de Imagens de Tomografia Computadorizada utilizando Redes Neurais Profundas e Transfer Learning</b>.

Pesquisa realizada por Pedro Benedicto de Melo Cardana, sob a supervisão de Wallace Casaca, na Universidade Estadual Paulista Júlio de Mesquita Filho, câmpus são José do Rio Preto.

---

<h2>Preparação do Ambiente</h2>
Clonar o repositório, e utilizar o anaconda para instalar as bibliotecas necesssárias:

```
git clone https://github.com/AzurePi/Detecao_Covid_VGG-19.git
conda env create -f environment.yml
```

<h2>Execução</h2>

O script `vgg_19.py` utiliza uma versão ligeriamente modificada da arquitetura de rede neural VGG-19 (descrita pela primeira vez [nesse trabalho](https://arxiv.org/abs/1409.1556)), para aprender a identificar lesões em imagens de tomografia
computadorizada pulmonar associadas à infecção da Covid-19. Utilizamos dois modelos: um deles é treinado a partir de pesos inicializados aleatoriamente, enquanto o outro faz uso da técnica de transfer learning a partir de um pré-treinamento
na base [ImageNet](https://ieeexplore.ieee.org/document/5206848). É possível adapatar o código para permitir várias iterações do treinamento, permitindo uma ampla gama de comparações entre as técnicas empregadas.
  
Os arquivos `datasets.py` e `resultados.py` contém funções auxiliares para o download e preparação dos datasets utilizados na pesquisa, e para a salvar os resultados do treinamento. Utilizamos dois datasets em nosso projeto: 
[<i>CT Scans for Covid-19 Classification</i>](<https://www.kaggle.com/datasets/azaemon/preprocessed-ct-scans-for-covid19>) e <i>SARS-CoV-2 CT-scan dataset: A large dataset of real patients CT scans for SARS-CoV-2 </i> (SOARES, E. et al.).
Ao final, os históricos dos processos de treinamento são salvos em logs, e resumidos em gráficos.
