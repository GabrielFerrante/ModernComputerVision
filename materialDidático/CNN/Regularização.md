# Regularização em Redes Neurais Convolucionais (CNNs)

Regularização é essencial para evitar *overfitting* em CNNs, garantindo que o modelo generalize bem para dados não vistos. Abaixo estão as principais técnicas aplicadas a CNNs:

---

## 1. **Regularização L1 e L2**
- **Descrição**: Penalizam pesos grandes adicionando termos à função de perda.
  - **L1 (Lasso)**: Adiciona \(\lambda \sum |w_i|\), promovendo esparsidade.
  - **L2 (Ridge)**: Adiciona \(\lambda \sum w_i^2\), limitando magnitudes.
- **Aplicação**:
  - Implementada via `kernel_regularizer` em camadas convolucionais.
- **Vantagens**:
  - L1 seleciona características mais relevantes.
  - L2 previne pesos extremos.
- **Cenários**:
  - L1 para seleção de features em CNNs com muitas camadas.
  - L2 como padrão para controle de magnitude.


 - **Nota**: A esparsidade refere-se à presença de valores zero (ou próximos de zero) nos parâmetros da rede (como pesos ou ativações), reduzindo conexões redundantes e simplificando o modelo. 
    - Esparsidade de pesos: Quando muitos pesos nas camadas convolucionais ou densas são próximos de zero, tornando partes da rede "inativas".

    - Esparsidade de ativações: Quando as saídas de neurônios ou filtros são zero (ex.: ReLU, que gera ativações esparsas).

---

## 2. **Dropout**
- **Descrição**: Desativa neurônios aleatoriamente durante o treino.
- **Variantes para CNNs**:
  - **Spatial Dropout**: Desativa canais inteiros (útil para feature maps 3D).
  - **DropBlock**: Remove blocos contíguos de unidades (eficaz para correlações espaciais).
- **Aplicação**:
  - `layers.Dropout(rate)` no TensorFlow/PyTorch.
- **Vantagens**:
  - Reduz dependências entre neurônios.
  - DropBlock é mais eficaz em camadas convolucionais.
- **Cenários**:
  - Spatial Dropout em camadas com alta correlação espacial (ex.: early layers).

---

## 3. **Data Augmentation**
- **Descrição**: Aumenta dados de treino via transformações.
- **Técnicas**:
  - **Clássicas**: Rotação, flip, escala, crop, ajustes de cor.
  - **Avançadas**:
    - **Cutout**: Mascara regiões da imagem.
    - **Mixup**: Combina pares de imagens e labels.
    - **CutMix**: Substitui regiões por partes de outras imagens.
- **Aplicação**:
  - Bibliotecas como `torchvision.transforms` ou `tf.image`.
- **Vantagens**:
  - Simula variações do mundo real.
  - Cutout/Mixup melhoram robustez a oclusões.
- **Cenários**:
  - Essencial em datasets pequenos (ex.: classificação médica).

---

## 4. **Early Stopping**
- **Descrição**: Interrompe o treino quando a loss de validação para de melhorar.
- **Implementação**:
  - Callbacks como `EarlyStopping` no Keras.
- **Vantagens**:
  - Evita overfitting sem alterar a arquitetura.
- **Cenários**:
  - Útil quando o tempo de treino é limitado.

---

## 5. **Batch Normalization (BN)**
- **Descrição**: Normaliza as saídas de cada camada usando média/variância do batch.
- **Efeito Regularizador**:
  - Introduz ruído estatístico via mini-batches.
- **Aplicação**:
  - `layers.BatchNormalization()` após camadas convolucionais.
- **Vantagens**:
  - Acelera convergência e reduz sensibilidade à inicialização.
- **Cenários**:
  - Quase universal em CNNs profundas (ex.: ResNet).

---

## 6. **Label Smoothing**
- **Descrição**: Substitui labels "hard" (0 ou 1) por valores suavizados (ex.: 0.1 e 0.9).
- **Aplicação**:
  - Parâmetro em funções de perda (ex.: `losses.CategoricalCrossentropy(label_smoothing=0.1)`).
- **Vantagens**:
  - Redui overconfidence do modelo.
- **Cenários**:
  - Classificação com labels ruidosas.

---

## 7. **Stochastic Depth**
- **Descrição**: Aleatoriamente "desliga" camadas durante o treino.
- **Implementação**:
  - Skipping aleatório de camadas em redes residuais (ex.: ResNet).
- **Vantagens**:
  - Reduz dependência hierárquica entre camadas.
- **Cenários**:
  - CNNs muito profundas (ex.: ResNet-1202).

---

## 8. **Transfer Learning & Fine-Tuning**
- **Descrição**: Reutiliza pesos pré-treinados em tarefas similares.
- **Aplicação**:
  - Congelar camadas iniciais e ajustar camadas finais.
- **Vantagens**:
  - Aproveita conhecimento de grandes datasets (ex.: ImageNet).
- **Cenários**:
  - Datasets pequenos ou similares ao dataset pré-treinado.

---

## 9. **Max-Norm Constraint**
- **Descrição**: Limita a norma máxima dos vetores de pesos.
- **Implementação**:
  - Restrição via `kernel_constraint` em camadas.
- **Vantagens**:
  - Controla crescimento dos pesos sem alterar a função de perda.
- **Cenários**:
  - Combinado com dropout ou L2.

---

## 10. **Noise Injection**
- **Tipos**:
  - **Input Noise**: Adiciona ruído Gaussiano às entradas.
  - **Weight Noise**: Perturba pesos durante o treino.
- **Efeito**:
  - Aumenta robustez a pequenas variações.
- **Cenários**:
  - Redes com pouca diversidade nos dados.

---

## Resumo Comparativo
| Técnica               | Vantagens                              | Melhor Uso                |
|-----------------------|----------------------------------------|---------------------------|
| L1/L2                 | Controle de magnitude/esparsidade      | Camadas densas/convolucionais |
| DropBlock             | Efetivo em correlações espaciais       | Camadas intermediárias    |
| Data Augmentation     | Aumenta diversidade artificialmente    | Datasets pequenos         |
| BatchNorm             | Estabiliza treino + ruído estatístico  | Quase todas as CNNs       |
| Label Smoothing       | Reduz overconfidence                   | Classificação com ruído   |
| Transfer Learning     | Reutiliza conhecimento prévio          | Tarefas com dados limitados |

---

**Conclusão**: A escolha depende do problema, tamanho do dataset e arquitetura. Combinações (ex.: **DropBlock + L2 + Data Augmentation**) são comuns para maximizar a generalização.