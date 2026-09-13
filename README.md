# Detecção de Fraudes em Transações Bancárias

Projeto de Machine Learning para identificar transações de cartão de crédito fraudulentas em um dataset real, altamente desbalanceado. O objetivo é encontrar padrões que indiquem se uma transação é fraudulenta ou não, comparando diferentes estratégias de balanceamento e algoritmos de classificação.

## 📊 Dataset

O projeto utiliza o dataset público **Credit Card Fraud Detection**, carregado diretamente via URL:

```
https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
```

As colunas `V1` a `V28` são features numéricas resultantes de uma transformação PCA (por questões de privacidade), além de `Time`, `Amount` e a variável alvo `Class` (0 = transação legítima, 1 = fraude).

### Desbalanceamento de classes

Apenas **0,17%** das transações do dataset são fraudulentas, contra **99,83%** de transações legítimas. Esse desbalanceamento é o principal desafio do projeto: um modelo que sempre prevê "não fraude" atingiria 99% de acurácia sem nenhuma utilidade prática, reforçando a necessidade de métricas como precisão, recall e AUC no lugar da acurácia simples.

## 🔧 Pipeline do projeto

1. **Carregamento e exploração dos dados**
2. **Feature Engineering**
   - Normalização logarítmica (`log1p`) da coluna `Amount`
   - Padronização com `StandardScaler`
3. **Divisão treino/teste** com `train_test_split` estratificado (mantendo a proporção de fraudes)
4. **Modelagem baseline** com Regressão Logística
5. **Avaliação** com `classification_report`, curva ROC (AUC) e curva Precision-Recall
6. **Balanceamento de dados**
   - Undersampling da classe majoritária
   - Oversampling sintético com SMOTE
7. **Modelos mais avançados**
   - Random Forest (`class_weight='balanced'`)
   - Ajuste de threshold de decisão em um pipeline com Regressão Logística
   - XGBoost (`scale_pos_weight` para lidar com o desbalanceamento)
8. **Importância das variáveis** (feature importance do XGBoost)
9. **Ajuste de hiperparâmetros** com `GridSearchCV` (otimizando para recall)
10. **Explicabilidade do modelo** com SHAP (gráfico de waterfall)

## 🤖 Modelos treinados

| Modelo | Precisão (fraude) | Recall (fraude) | F1-score (fraude) | AUC |
|---|---|---|---|---|
| Regressão Logística | 0.85 | 0.65 | 0.74 | 0.95 |
| Random Forest | 0.83 | 0.76 | 0.80 | — |
| Regressão Logística (threshold ajustado) | 0.79 | 0.69 | 0.74 | — |
| **XGBoost** | **0.94** | **0.78** | **0.85** | — |

O **XGBoost** apresentou o melhor equilíbrio entre precisão e recall, sendo o modelo com melhor desempenho geral entre os testados.

## 📈 Métricas utilizadas

Por se tratar de um problema de classificação desbalanceada, a acurácia não é um bom indicador de desempenho. O projeto prioriza:

- **Precisão**: das transações marcadas como fraude, quantas realmente eram
- **Recall**: das fraudes reais, quantas foram identificadas
- **F1-score**: equilíbrio entre precisão e recall
- **AUC-ROC**: capacidade do modelo de distinguir as duas classes
- **Curva Precision-Recall**: análise do trade-off entre precisão e recall conforme o threshold de decisão varia

## 🧠 Explicabilidade (SHAP)

Para tornar o modelo XGBoost interpretável, foi utilizado o **SHAP (SHapley Additive exPlanations)**, permitindo visualizar quais variáveis mais influenciam a decisão do modelo para cada transação individual.

## 🛠️ Tecnologias utilizadas

- Python
- Pandas / NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost
- Matplotlib
- SHAP

## ▶️ Como executar

O notebook foi desenvolvido no Google Colab. Para executá-lo:

1. Abra o notebook `Detecção_de_Anomalias_em_transações.ipynb` no [Google Colab](https://colab.research.google.com/)
2. Instale as dependências que não vêm por padrão no ambiente:

```bash
pip install imbalanced-learn xgboost shap
```

3. Execute as células em ordem — o dataset é carregado automaticamente via URL, não sendo necessário upload manual.

## 📌 Próximos passos (sugestões)

- Validação cruzada mais robusta para todos os modelos
- Testar outras técnicas de balanceamento (ex: SMOTE + Tomek Links)
- Deploy do modelo como API para inferência em tempo real
- Monitoramento de drift do modelo em produção
