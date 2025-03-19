# Redes Neurais Convolucionais (CNNs)

## O que são Redes Neurais Convolucionais?

Redes Neurais Convolucionais (CNNs, do inglês *Convolutional Neural Networks*) são um tipo de arquitetura de rede neural profunda projetada especificamente para processar dados com estrutura espacial, como imagens, vídeos e sinais multidimensionais. Elas são amplamente utilizadas em tarefas de visão computacional, reconhecimento de padrões e classificação de imagens.

### Características Principais:

1. **Convolutions (Convoluções)**
A operação de convolução é a espinha dorsal das CNNs. Consiste em deslizar um kernel (filtro) sobre a entrada para produzir um feature map.

    - Matematicamente: A equação para calcular o valor de um pixel no feature map de saída é:

    ![alt text](image-1.png)
    
    O termo I (Input Image/Image de Entrada):
    - O que é: Representa a imagem de entrada, geralmente uma matriz 2D (grayscale) ou 3D (colorida, com canais RGB).

    -   Formato: ![alt text](image-2.png)

    O termo K (Kernel/Filtro): 
    - O que é: Uma matriz pequena (ex: 3×3) de pesos aprendidos durante o treinamento. 
    - Formato: K é uma matrix F X F, F é o tamanho do kernel
    - ![alt text](image-3.png)

    O termo i,j (Coordenadas no Feature Map de Saída):
    - O que são: Índices que indicam a posição do pixel no feature map resultante.
    - ![alt text](image-4.png)

    O termo m,n (Índices do Kernel):
    - O que são: Índices que percorrem as linhas (m) e colunas (n) do kernel.
    - ![alt text](image-5.png)

    O termo I(i+m,j+n) (Região da Imagem Sob o Kernel):
    - O que é: A submatriz da imagem I coberta pelo kernel na posição atual.
    - Fornece os pixels que interagem com o kernel para gerar o valor de saída.

    O termo ⋅ (Multiplicação Elemento a Elemento)
    - O que é: Operação de multiplicação entre cada pixel da região da imagem (I(i+m,j+n)) e o peso correspondente do kernel (K(m,n)).

    ![alt text](image-6.png)

    - Nota: Canais de Cor (RGB): Em imagens coloridas, a equação inclui uma soma adicional sobre os canais


    ![alt text](image-7.png)

2. **Kernel Size and Depth (Tamanho e Profundidade do Kernel)**

- Os kernels podem ter varios de tamanhos.
- Geralmente de tamanho impar, pois possuem mais simetria entre pontos centrais da janela. 
- Tamanho do mapa de características resultante : Tamanho da entrada - Tamanho do Kernel + 1

- Kernels menores:

    - Capturam detalhes locais.

    - Reduzem parâmetros (eficientes computacionalmente).

- Kernels maiores:

    - Capturam contexto global.

    - Úteis em camadas iniciais para downsampling.

- Profundidade (D):

    - Número de kernels em uma camada.

    - Define quantos tipos de features são extraídos.

    Exemplo: 64 kernels → 64 feature maps distintos.


3. **Padding**

- Adição de zeros ou valores ao redor da entrada para controlar o tamanho da saída. 
- O objetivo do padding é manter o tamanho da saída do feature map igual ao da imagem de entrada. Para isso, a adição de zeros envolta da entrada é necessária. 

- Para redes muito profundas, não queremos continuar reduzindo o tamanho

- Os pixels nas bordas contribuem menos para os mapas de características de saída, portanto, estamos descartando informações deles.

![alt text](image-8.png)

Nota: A fórmula é aplicada para cada dimensão do mapa final.


4. **Stride**

- Define quantos pixels o kernel se move a cada passo.
- basicamente, o tamanho do passo define o tamanho da saída. Com stride 1, não há redução de tamanho, porém um stride 2 (cada passo pular 2 linhas e 2 colunas na matrix) reduz pela metade o tamanho.

- Efeitos:
    - Stride S = 1: Sobreposição máxima (maior precisão).
    - Stride S = 2: Reduz a resolução pela metade (downsampling).

- Exemplo:

    - Entrada 5×5, kernel 3×3, stride 2:
    - Saída 2×2 (sem padding).

- Aplicação:
    - Substitui camadas de pooling em algumas arquiteturas (ex: Conv com stride 2).