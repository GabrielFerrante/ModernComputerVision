# Avaliação de Modelos: Métricas e Interpretação

A avaliação de modelos de **Redes Neurais Convolucionais (CNNs)** é crucial para entender seu desempenho em tarefas de classificação. Abaixo estão as principais métricas utilizadas, suas fórmulas, interpretações e exemplos de aplicação.

---

## 1. Matriz de Confusão
A **matriz de confusão** é uma tabela que resume os resultados de classificação de um modelo, comparando as previsões com os valores reais. Ela organiza os resultados em quatro categorias:

- **Verdadeiros Positivos (TP):** Casos onde o modelo previu corretamente a classe positiva.
- **Verdadeiros Negativos (TN):** Casos onde o modelo previu corretamente a classe negativa.
- **Falsos Positivos (FP):** Casos onde o modelo previu positivo incorretamente (erro Tipo I).
- **Falsos Negativos (FN):** Casos onde o modelo previu negativo incorretamente (erro Tipo II).

### Exemplo:
|                 | Previsto Positivo | Previsto Negativo |
|-----------------|-------------------|-------------------|
| **Real Positivo** | TP = 80           | FN = 10           |
| **Real Negativo** | FP = 5            | TN = 95           |

### Interpretação:
- A matriz permite visualizar **onde o modelo erra** e quais classes são mais confundidas.
- Base para calcular outras métricas como precisão, recall e F1-Score.

---

## 2. Acurácia
A **acurácia** mede a proporção de previsões corretas (tanto positivas quanto negativas) em relação ao total.

\[
\text{Acurácia} = \frac{TP + TN}{TP + TN + FP + FN}
\]

### Exemplo:
No cenário acima:  
\[
\text{Acurácia} = \frac{80 + 95}{80 + 95 + 5 + 10} = \frac{175}{190} \approx 92.1\%
\]

### Interpretação:
- Útil quando as classes estão **balanceadas**.
- **Limitação:** Pode ser enganosa em conjuntos desbalanceados. Por exemplo, se 95% dos dados são negativos, um modelo que sempre prevê negativo terá 95% de acurácia, mas não é útil.

---

## 3. Precisão
A **precisão** indica a proporção de previsões positivas corretas em relação a todas as previsões positivas.

\[
\text{Precisão} = \frac{TP}{TP + FP}
\]

### Exemplo:
\[
\text{Precisão} = \frac{80}{80 + 5} = 94.1\%
\]

### Interpretação:
- Foco em **minimizar falsos positivos**.
- Ideal para cenários onde custos de FP são altos (ex.: classificar e-mails como spam incorretamente).

---

## 4. Recall (Sensibilidade)
O **recall** mede a capacidade do modelo de identificar corretamente os casos positivos.

\[
\text{Recall} = \frac{TP}{TP + FN}
\]

### Exemplo:
\[
\text{Recall} = \frac{80}{80 + 10} = 88.9\%
\]

### Interpretação:
- Foco em **minimizar falsos negativos**.
- Importante em diagnósticos médicos (ex.: não deixar pacientes doentes sem tratamento).

---

## 5. F1-Score
O **F1-Score** é a média harmônica entre precisão e recall, equilibrando as duas métricas.

\[
\text{F1-Score} = 2 \times \frac{\text{Precisão} \times \text{Recall}}{\text{Precisão} + \text{Recall}}
\]

### Exemplo:
\[
\text{F1-Score} = 2 \times \frac{0.941 \times 0.889}{0.941 + 0.889} \approx 91.4\%
\]

### Interpretação:
- Útil quando há **desequilíbrio entre precisão e recall**.
- Ideal para problemas onde ambos FP e FN são críticos (ex.: detecção de fraudes).

---

## Como Escolher a Métrica Certa?
| Métrica       | Quando Usar?                                                                 |
|---------------|-----------------------------------------------------------------------------|
| **Acurácia**  | Classes balanceadas e custos de erros similares.                            |
| **Precisão**  | Custos de FP são altos (ex.: recomendação de produtos).                     |
| **Recall**    | Custos de FN são altos (ex.: diagnóstico de doenças).                       |
| **F1-Score**  | Balancear precisão e recall (ex.: classificação de textos com múltiplas classes). |

---

## Conclusão
- A **matriz de confusão** é a base para entender o desempenho do modelo.
- **Acurácia** é intuitiva, mas falha em conjuntos desbalanceados.
- **Precisão** e **Recall** são complementares: priorize a primeira para evitar FP e a segunda para evitar FN.
- **F1-Score** é uma métrica robusta para cenários que exigem equilíbrio.
- A escolha da métrica depende do **contexto do problema** e dos **custos associados a erros**.