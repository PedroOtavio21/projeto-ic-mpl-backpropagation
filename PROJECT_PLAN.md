# PROJECT_PLAN.md — MLP Dry Bean

**Versão:** 3.0 — execução até 07/11/2026  
**Prazo:** sábado, 07/11/2026  
**Equipe:** 2 integrantes + OpenCode/DeepSeek Flash  
**Fonte de verdade:** este arquivo.

---

## 1. Objetivo e escopo

Implementar uma MLP com Backpropagation para classificar as 7 variedades do Dry Bean Dataset, cobrindo integralmente as questões Q1–Q7, com resultados reproduzíveis e relatório em Markdown.

**Escopo:**
- Código exclusivamente em notebooks `.ipynb`.
- Um notebook de experimentos e um de análise.
- CSVs, figuras e tabelas em `results/`.
- Relatório final em `report/relatorio.md`.

**Fora de escopo:** `.py` como código-fonte, GUI, API, deploy, Docker, CI/CD, AutoML, modelos adicionais, regularização, dropout, class weights, data augmentation, múltiplas seeds em Q2–Q6 e experimentos não exigidos.

---

## 2. Requisitos das sete questões

| Questão | Requisito | Implementação | Evidência |
|---|---|---|---|
| Q1 | 5 inicializações, ReLU/Softmax, loss/acurácia/curvas, somente treino | 16-60-7, 5 seeds, 100 épocas | `q1_runs.csv`, `q1_history.csv`, figuras |
| Q2 | Grid de learning rate e momentum, convergência ≤ 0,001, épocas, melhor combinação | 9 combinações, treino + validação | `q2_grid.csv`, heatmaps |
| Q3 | 1–3 camadas, 10–100 neurônios em passos de 10, treino/validação, tempo, F1, top 4 | 30 configurações com largura uniforme | `q3_grid.csv`, top 4 |
| Q4 | 20%, 40%, 60%, 80%, 100% do treino, estratificado, loss/acurácia/F1/tempo, curvas | 5 frações, validação fixa | `q4_fractions.csv`, curvas |
| Q5 | Conjunto completo + redundantes, irrelevantes e ruidosos, comparação, melhor conjunto | 5 regras objetivas, ajuste só no treino | `q5_features.csv`, `q5_subsets.csv` |
| Q6 | 4 configurações, teste, comparação, uma vencedora, curvas | Top 4 da Q3 + Q2 + Q5, treino no desenvolvimento de 70% | `q6_test.csv`, matriz de confusão |
| Q7 | Validação cruzada k-fold, média e variância, curvas por partição, robustez | k=5, pré-processamento dentro de cada fold | `q7_cv.csv`, `q7_summary.csv` |

---

## 3. Tecnologias e dependências

| Finalidade | Ferramenta |
|---|---|
| Linguagem | Python 3.11 |
| Ambiente | Jupyter Notebook / Google Colab |
| Dados | Pandas, NumPy |
| Pré-processamento | Scikit-learn |
| Rede neural | TensorFlow/Keras |
| Visualização | Matplotlib, Seaborn |
| Versionamento | Git, GitHub |
| Relatório | Markdown |

`requirements.txt`:

```text
numpy==1.26.4
pandas==2.1.4
scikit-learn==1.3.2
tensorflow-cpu==2.15.1
matplotlib==3.8.2
seaborn==0.13.0
jupyter==1.0.0
```

Em ambiente local foi usado `tensorflow-cpu` para reduzir o custo computacional; em GPU/Colab, `tensorflow==2.15.1` é equivalente.

---

## 4. Arquitetura e estrutura de diretórios

```text
mlp-dry-bean-backprop/
├── README.md
├── AGENTS.md
├── PROJECT_PLAN.md
├── requirements.txt
├── .gitignore
├── data/
│   └── raw/
│       └── Dry_Bean_Dataset.csv
├── notebooks/
│   ├── 01_experimentos.ipynb
│   └── 02_analise_resultados.ipynb
├── results/
│   ├── csv/
│   ├── figures/
│   └── tables/
└── report/
    └── relatorio.md
```

| Caminho | Responsabilidade |
|---|---|
| `01_experimentos.ipynb` | Dados, split, funções, Q1–Q7, CSVs e figuras. |
| `02_analise_resultados.ipynb` | Apenas lê CSVs; gera tabelas Markdown e figuras consolidadas. |
| `results/csv/` | Resultados e históricos. |
| `results/figures/` | Figuras PNG. |
| `results/tables/` | Tabelas Markdown. |
| `report/relatorio.md` | Relatório final. |

---

## 5. Organização dos notebooks

**`01_experimentos.ipynb`:** imports/configuração, seeds, dados, EDA mínima, split 70/30 com validação interna 80/20, funções reutilizáveis, Q1–Q7, `winners.csv`.

**`02_analise_resultados.ipynb`:** carrega os CSVs, gera `results/tables/*.md` e figuras finais. Não treina.

Cada questão tem células Markdown de configuração, execução, tabela, figuras e análise. O notebook deve rodar de cima para baixo.

**Propriedade:** apenas um integrante edita `01_experimentos.ipynb` por vez. A implementa Q1–Q3; B assume Q4–Q7. O notebook de análise e o relatório podem ser trabalhados em paralelo.

---

## 6. Protocolo experimental

### 6.1 Dados e split
- Split externo estratificado: 70% desenvolvimento, 30% teste, seed 42.
- Split interno do desenvolvimento: 80% treino, 20% validação → efetivo **56% treino / 14% validação / 30% teste**.
- Q1–Q5 usam 56% treino e 14% validação; Q1 usa apenas os 56% de treino.
- Q4 usa frações de 20–100% dos 56% de treino.
- Q6 retreina no desenvolvimento completo (70%) e avalia no teste (30%).
- Q7 faz CV k=5 no desenvolvimento (70%); o teste permanece intocado.
- `StandardScaler` ajustado somente no treino correspondente; one-hot nas classes.
- Macro-F1 é a métrica principal.

### 6.2 Modelo
- Entrada: 16 atributos (ou subconjunto da Q5).
- Ocultas: ReLU, Glorot uniform, vieses zero.
- Saída: 7 neurônios Softmax.
- Perda: entropia cruzada categórica.
- Otimizador: SGD com momento, sem Nesterov.
- Batch: 32.
- Sem dropout, regularização, class weights ou early stopping na validação.

### 6.3 Treinamento e convergência
- Q1: 100 épocas fixas.
- Q2–Q7: máximo de 150 épocas; parada imediata quando a loss de treino ≤ 0,001.
- Se não convergir: `converged=False`, `epochs=150`; não relaxar o limiar.
- Em NaN/divergência: `status="diverged"` e continuar o grid.
- Seeds: Q1 = 42–46; Q2–Q6 = 42; Q7 = folds com seed 42.

### 6.4 Uso do teste
- Teste intocado em Q1–Q5.
- Q6: avaliação final e seleção da vencedora.
- Q7: CV no desenvolvimento de 70%; teste intocado.
- Em Q6, as curvas de teste são registradas por callback apenas para visualização; nenhum callback de parada, restauração de pesos ou seleção de hiperparâmetros lê essas métricas.

### 6.5 Questões

**Q1 — Inicialização:** topologia 16-60-7; SGD lr=0,01, momento=0,9; 100 épocas; seeds 42–46; somente treino.

**Q2 — Learning rate e momento:** 9 combinações: lr ∈ {0,01; 0,1; 0,5} × momento ∈ {0,5; 0,7; 0,9}; topologia da Q1; máximo 150 épocas; treino + validação. Seleção: entre convergidos, maior macro-F1 de validação; candidatos dentro de 0,01; menor número de épocas; desempate por menor loss de validação.

**Q3 — Topologia:** 1–3 camadas; 10–100 neurônios, passo 10; largura uniforme; 30 configurações; hiperparâmetros da Q2; máximo 150 épocas; treino + validação. Top 4 por macro-F1 de validação; desempate por menor loss de validação, menor tempo e menor gap.

**Q4 — Quantidade de dados:** melhor configuração da Q3, todos os 16 atributos; frações 20%, 40%, 60%, 80%, 100% dos 56% de treino; estratificado; scaler ajustado no subconjunto; validação fixa; máximo 150 épocas. Seleção: maior macro-F1 de validação; se 100% estiver dentro de 0,005 do melhor, usar 100%. A Q4 é uma análise de sensibilidade e não altera os dados usados na Q5/Q6.

**Q5 — Atributos:** melhor configuração da Q3 + hiperparâmetros da Q2; treino/validação de 56%/14%. Cálculos somente no treino: informação mútua (MI), correlação de Pearson e variância. Subconjuntos:
1. `all_16`.
2. `low_redundancy`: grafo de pares com |r| ≥ 0,95; em cada componente conexa, manter o maior MI e remover os demais.
3. `top_10_mi`: 10 maiores MI (desempate alfabético).
4. `compact`: `low_redundancy` e depois top-10 MI restantes.
5. `no_noise`: remover variância ≤ percentil 10 e MI ≤ mediana; se nenhum for removido, reutilizar `all_16`.

Seleção: maior macro-F1 de validação; candidatos dentro de 0,005; menor número de atributos; desempate por menor loss e menor tempo.

**Q6 — Avaliação final:** top 4 topologias da Q3; hiperparâmetros da Q2; regra de atributos da Q5 reajustada no desenvolvimento (70%); treino no desenvolvimento; teste (30%). Máximo 150 épocas; parada pela loss de treino. Métricas de treino e teste; callback de teste apenas descritivo. Vencedora por macro-F1 de teste; desempate por weighted-F1, loss de teste e tempo.

**Q7 — Validação cruzada:** k=5 estratificado no desenvolvimento (70%), shuffle=True, seed 42; scaler e regra de atributos reajustados em cada fold; configuração vencedora da Q6; máximo 150 épocas. Reportar métricas por fold, média, variância, desvio-padrão, mínimo e máximo. O teste de 30% não é usado.

---

## 7. Métricas e critérios de seleção

| Métrica | Uso |
|---|---|
| Loss (entropia cruzada) | Treinamento, validação e teste; critério de parada. |
| Acurácia | Comparação geral. |
| Macro-F1 | Métrica principal de seleção. |
| Weighted-F1 | Complementar. |
| Precisão/revogação macro | Q6 e Q7. |
| Tempo, épocas, parâmetros | Q3–Q7. |

Critérios de seleção definidos na seção 6.5, aplicados sem ajuste após ver os resultados. Nenhum resultado será inventado.

---

## 8. Organização dos resultados

| Arquivo | Conteúdo |
|---|---|
| `split_indices.csv` | Índice e partição de cada instância. |
| `class_mapping.csv` | Índice e nome das classes. |
| `q1_runs.csv`, `q1_history.csv` | Q1. |
| `q2_grid.csv`, `q2_history.csv` | Q2. |
| `q3_grid.csv`, `q3_history.csv` | Q3 e ranking. |
| `q4_fractions.csv`, `q4_history.csv` | Q4. |
| `q5_features.csv`, `q5_subsets.csv`, `q5_history.csv` | Q5. |
| `q6_test.csv`, `q6_history.csv` | Q6. |
| `q7_folds.csv`, `q7_cv.csv`, `q7_history.csv`, `q7_summary.csv` | Q7. |
| `winners.csv` | Vencedor de cada etapa. |

Figuras em `results/figures/`; tabelas Markdown em `results/tables/`. Gravar CSV após cada execução; não salvar pesos.

---

## 9. Estratégia de execução e cache

- Ordem: dados → Q1 → Q2 → Q3 → Q4 → Q5 → Q6 → Q7 → análise → relatório.
- Flags `FORCE_RERUN_QX`: se o CSV existir e a flag estiver desligada, carregar e não retreinar.
- `verbose=0`; imprimir uma linha-resumo por execução.
- `tf.keras.backend.clear_session()` entre execuções.
- Erros em uma configuração não abortam o grid; registrar status e continuar.
- Executar o notebook sempre de cima para baixo.
- Tempo computacional estimado: 2–5 h; Q3 é o gargalo.

---

## 10. Divisão das atividades entre os integrantes

| Integrante | Responsabilidade principal |
|---|---|
| **A** | Dados, split, funções, Q1–Q3, revisão de Q4–Q7, relatório Q1–Q3. |
| **B** | Notebook de análise, Q4–Q7, relatório Q4–Q7, consolidação final. |

Em paralelo: B prepara análise/template do relatório enquanto A executa Q1–Q3; após o núcleo, B assume Q4–Q7 enquanto A revisa e escreve Q1–Q3. Ninguém edita `01_experimentos.ipynb` ao mesmo tempo.

---

## 11. Cronograma até sábado

| Bloco | Janela | Atividade | Responsável | Conclusão |
|---|---|---|---|---|
| 1 | Qui 05/11 | Estrutura, dados, split, funções, Q1–Q2 | A | `q1_*` e `q2_*` CSVs |
| 2 | Sex 06/11 | Q3 e top 4 | A | `q3_grid.csv` |
| 3 | Sex 06/11 | Q4–Q5 | B | `q4_*` e `q5_*` CSVs |
| 4 | Sex 06/11 | Q6–Q7 | B | `q6_*` e `q7_*` CSVs |
| 5 | Sáb 07/11 | Análise, relatório, reprodutibilidade, tag | A+B | `relatorio.md` + `v1.0-entrega` |

**Prioridades:** P0 = Q1–Q7, CSVs, política de teste, relatório; P1 = figuras/tabelas, análise, README, reprodutibilidade; P2 = EDA extra, métricas por classe, experimentos adicionais.

**Contingência:** se a Q3 exceder o tempo, reduzir apenas o máximo de épocas (não o número de configurações) e registrar as não convergências. Se necessário, dividir as 30 configurações em dois blocos manuais.

---

## 12. Checklist de entrega

- [ ] Q1: 5 seeds, ReLU/Softmax, curvas e tabela.
- [ ] Q2: 9 combinações, convergência ≤ 0,001, épocas e melhor combinação.
- [ ] Q3: 30 topologias, treino/validação, tempo, F1 e top 4.
- [ ] Q4: 20/40/60/80/100%, estratificado, métricas e curvas.
- [ ] Q5: conjunto completo + 4 subconjuntos, critérios objetivos e melhor conjunto.
- [ ] Q6: 4 configurações, teste, curvas e vencedora.
- [ ] Q7: k=5, média/variância, curvas por fold e limitação registrada.
- [ ] Teste não usado em Q1–Q5 e Q7.
- [ ] CSVs, figuras e tabelas gerados.
- [ ] Relatório Markdown com todas as seções e análises.
- [ ] `requirements.txt`, README e reprodutibilidade.
- [ ] Git organizado e tag final.
