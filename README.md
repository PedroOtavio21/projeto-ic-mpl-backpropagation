# MLP Dry Bean — Backpropagation

Classificação multiclasse das 7 variedades do Dry Bean Dataset com uma MLP treinada por Backpropagation.

## Estrutura

```text
data/raw/Dry_Bean_Dataset.csv   # dataset (13.611 x 17)
notebooks/01_experimentos.ipynb # Q1–Q7
notebooks/02_analise_resultados.ipynb
results/csv/                    # resultados e históricos
results/figures/                # figuras
results/tables/                 # tabelas Markdown
report/relatorio.md             # relatório técnico
```

## Ambiente

Requer Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No Colab, use `tensorflow==2.15.1` em vez de `tensorflow-cpu`.

## Ordem de execução

1. `notebooks/01_experimentos.ipynb` — carrega os dados, executa Q1–Q7 e grava CSVs/figuras.
2. `notebooks/02_analise_resultados.ipynb` — lê os CSVs e gera tabelas/figuras finais.
3. `report/relatorio.md` — relatório com os resultados efetivamente obtidos.

Os notebooks usam cache por questão (`FORCE_RERUN_QX`). Para refazer uma questão, apague o CSV correspondente ou ligue a flag.

## Protocolo resumido

- Split externo 70% desenvolvimento / 30% teste, estratificado, seed 42.
- Validação interna de 20% do desenvolvimento: 56% treino, 14% validação, 30% teste.
- Q1–Q5 usam treino/validação; Q6 usa o desenvolvimento completo e o teste; Q7 faz 5-fold no desenvolvimento.
- MLP com ReLU, Softmax, entropia cruzada categórica e SGD com momento.
- Critério de convergência: loss de treino ≤ 0,001.
- Métrica principal: macro-F1.

## Reprodutibilidade

- Seeds: Q1 42–46; Q2–Q6 42; Q7 folds com seed 42.
- Versões em `requirements.txt`.
- Splits e folds salvos em `results/csv/`.
- Nenhum peso de modelo é versionado.

## Fonte dos dados

UCI Machine Learning Repository — Dry Bean Dataset (ID 602). O CSV em `data/raw/` foi convertido do ARFF original, mantendo as 16 features e o rótulo `Class`.
