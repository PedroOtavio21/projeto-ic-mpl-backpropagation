# Classificação de Grãos de Feijão Seco com MLP Treinada por Backpropagation

> **Status:** template inicial. As seções de resultados serão preenchidas exclusivamente com os valores de `results/csv/` após a execução de `notebooks/01_experimentos.ipynb` e `notebooks/02_analise_resultados.ipynb`. Nenhum resultado fictício deve ser inserido.

## 1. Introdução e contextualização

Descrever o problema de classificação multiclasse, a MLP com Backpropagation, o Dry Bean Dataset e os objetivos das sete etapas.

## 2. Dataset e metodologia

- 13.611 instâncias, 16 atributos numéricos contínuos e 7 classes.
- Split externo 70% desenvolvimento / 30% teste, estratificado, seed 42.
- Validação interna de 20% do desenvolvimento: 56% treino / 14% validação / 30% teste.
- Normalização Z-score ajustada somente no treino; one-hot nas classes.
- MLP com ReLU nas ocultas e Softmax na saída; entropia cruzada categórica; SGD com momento.
- Macro-F1 como métrica principal; weighted-F1 como complementar.
- Critério de convergência: loss de treino ≤ 0,001, com máximo de 400 épocas.

## 3. Configurações experimentais

Preencher com versões de Python/TensorFlow, hardware, seeds, batch size, topologia-base e regras de seleção.

## 4. Resultados e discussão — Q1

Inserir configuração, tabela `results/csv/q1_runs.csv`, figuras `q1_loss_curves.png` e `q1_accuracy_curves.png`, análise de estabilidade e conclusão da etapa.

## 5. Resultados e discussão — Q2

Inserir grid `q2_grid.csv`, heatmaps e curvas, análise do efeito de learning rate/momento e conclusão.

## 6. Resultados e discussão — Q3

Inserir tabela das 30 topologias, tabela das 4 melhores, heatmaps e curvas; discutir complexidade, overfitting/underfitting e tempo; concluir.

## 7. Resultados e discussão — Q4

Inserir tabela das frações, curvas de generalização e tempo; discutir saturação e custo; concluir.

## 8. Resultados e discussão — Q5

Inserir tabela de atributos (MI, variância, correlação), tabela de subconjuntos e figuras; discutir redundância, irrelevância e ruído; concluir.

## 9. Resultados e discussão — Q6

Inserir tabela das quatro configurações, curvas de treino/teste e matriz de confusão; comparar generalização e selecionar a vencedora; concluir.

## 10. Resultados e discussão — Q7

Inserir tabela por fold, resumo com média/variância, curvas e boxplots; discutir robustez e a limitação de a Q6 ter usado o teste; concluir.

## 11. Conclusão geral

Sintetizar os achados das sete etapas, a configuração vencedora, o desempenho no teste e a robustez na validação cruzada, além das limitações.

## Referências

- UCI Machine Learning Repository. Dry Bean Dataset, ID 602.
- Documentação do TensorFlow/Keras e do Scikit-learn.
- KOKLU, M.; OZKAN, I. A. Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques. Computers and Electronics in Agriculture, 2020.
