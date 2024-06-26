# Processamento Digital de Imagens
Este repositório contém os códigos desenvolvidos na disciplina de Processamento Digital de Imagens (DCA0445).

## Unidade 1

### Und1 - Regions
Neste exercício, o programa solicita ao usuário as coordenadas de dois pontos localizados dentro dos limites do tamanho da imagem, onde será exibido o negativo da imagem na região do retângulo definido pela localização dos pontos. A imagem gerada pode ser vista abaixo:

<p align="center">
 <img src="https://github.com/RychardRS/Processamento-Digital-de-Imagens/assets/93292522/b7768f57-331a-4cdb-9a3e-0fd760b8de80" alt="biel">
 <img src="https://github.com/RychardRS/Processamento-Digital-de-Imagens/assets/93292522/f9530e19-e910-459d-bd61-bb483c584003" alt="bielNegativo">
</p>

### Und1 - Trocar Regiões
Neste exercício, o programa deverá trocar os quadrantes em diagonal na imagem, assuma que a imagem de entrada tem dimensões múltiplas de 2 para facilitar a implementação do processo de troca, na imagem abaixo, podemos ver no lado esquerdo a imagem original e na direita o resultado da execução do programa:

<p align="center">
 <img src="https://github.com/RychardRS/Processamento-Digital-de-Imagens/assets/93292522/94cf6df2-7f18-4bf9-958f-7128d4e251ba" alt="troca-regioes-resultado">
</p>

### Und1 - File Storage
Nesta prática, devemos criar um programa que gere uma imagem de dimensões 256x256 pixels contendo uma senóide de 4 períodos com amplitude igual 127 desenhada na horizontal, os resultados podem ser vistos abaixo:

<p align="center">
 <img src="https://github.com/RychardRS/Processamento-Digital-de-Imagens/assets/93292522/de6fdbcf-7469-49e7-a308-2f50b95e49af" alt="senoide1-256">
 <img src="https://github.com/RychardRS/Processamento-Digital-de-Imagens/assets/93292522/4b0fe4a6-36a4-4dd0-90f6-dc2a1e21d1ee" alt="senoide2-256">
 <img src="https://github.com/RychardRS/Processamento-Digital-de-Imagens/assets/93292522/346e10f3-0efe-4ac8-84bc-f5d80f75520f" alt="result">
</p>

- Temos respectivamente a senoide para 8 períodos, 4 períodos e o resultado, observando o comportamento da imagem, podemos concluir que no resultado final:
    1. Quando os pixels são brancos ou pretos nas duas imagens, o resultado será a cor preta.
    2. Caso tenha cor preta em uma imagem, mas branca em outra, o resultado será cor branca.
    3. Caso tenha locais em que há transição das cores, o resultado será uma cor acinzentada.

### Und1 - Esteganografia
Nesta atividade devemos escrever um programa que recupere a imagem codificada de uma imagem resultante de esteganografia, abaixo, será exibido na esquerda a imagem original e na direita a imagem decodificada.

<p align="center">
 <img src="https://github.com/RychardRS/Processamento-Digital-de-Imagens/assets/93292522/dc9fe18f-e248-480b-9630-08119f73c22a" alt="desafio-esteganografia">
 <img src="https://github.com/RychardRS/Processamento-Digital-de-Imagens/assets/93292522/edda3c48-eed1-4cad-bcac-2fbae300d3da" alt="decoded_output">
</p>

### Und1 - Labeling
Nesta prática, é possível verificar que caso existam mais de 255 objetos na cena, o processo de rotulação poderá ficar comprometido. Devemos identificar a situação em que isso ocorre e propor uma solução para este problema.

O motivo que isso acontece é porque quando o programa se limita a usar 256 tons de cinza, indo de 0 a 255, o programa se baseia na contagem de objetos na imagem, para poder pintar, então, se existir mais de 255 objetos na imagem, o programa ficará comprometido. A solução que podemos utilizar para resolver esse problema, é realizar a leitura da imagem de forma colorida (RGB), para que tenhamos uma maior variedade de valores para contagem.

Após utilizar o código de aprimoramento, do arquivo:
https://github.com/RychardRS/Processamento-Digital-de-Imagens/blob/main/Unidade%201%20-%20PDI/Labelling/labelling.cpp 

Abaixo temos a imagem original na esquerda, no meio a imagem sem bordas e na direita, a imagem final após a execução, tendo as bolhas com buraco pintadas em cinza mais escuro e as bolhas sem buracos em um cinza mais claro:

<p align="center">
 <img src="https://github.com/RychardRS/Processamento-Digital-de-Imagens/assets/93292522/a9012454-ff17-4462-8653-ddd12bc67f1e" alt="bolhas">
 <img src="https://github.com/RychardRS/Processamento-Digital-de-Imagens/assets/93292522/e19e7cb3-fe86-4e9c-bfb3-a7e32d257eb6" alt="nova-img-bolhas">
 <img src="https://github.com/RychardRS/Processamento-Digital-de-Imagens/assets/93292522/b5297157-b3f9-454b-a899-4e4dae299e11" alt="labelling">
</p>
