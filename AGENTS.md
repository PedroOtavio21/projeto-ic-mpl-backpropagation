# AGENTS.md — Instruções para agentes OpenCode

## Contexto obrigatório

Antes de implementar:
1. Ler `PROJECT_PLAN.md`.
2. Inspecionar os notebooks existentes.
3. Identificar a seção da questão afetada.
4. Respeitar arquitetura, escopo e decisões metodológicas.
5. Não redefinir a solução nem adicionar funcionalidades não solicitadas.

Se o pedido conflitar com `PROJECT_PLAN.md`, interromper e pedir esclarecimento.

---

## Código

- Python 3.11.
- Código exclusivamente em notebooks `.ipynb`; nunca criar `.py` como código-fonte.
- `snake_case` para variáveis e funções.
- Funções pequenas, com responsabilidade única e docstring objetiva.
- Reutilizar funções existentes; evitar duplicação.
- Organizar o notebook com seções Markdown e células de configuração, execução, resultados e análise.
- Evitar estado oculto; o notebook deve rodar de cima para baixo.
- Não criar abstrações, classes ou camadas de engenharia desnecessárias.
- Não adicionar bibliotecas sem justificativa aprovada.
- Não mover código para arquivos externos.

## Machine Learning

- Fixar seeds de Python, NumPy e TensorFlow.
- Garantir reprodutibilidade; registrar versões e configurações.
- Split antes de qualquer ajuste de scaler/seletor.
- Ajustar scaler e seleção de atributos somente no treinamento.
- Nunca usar o teste em Q1–Q5 e Q7.
- Em Q6, o teste só pode ser avaliado após o treinamento ou por callback descritivo; nunca para parada, restauração de pesos ou seleção de hiperparâmetros.
- Preservar o critério de convergência: loss de treino ≤ 0,001.
- Não inventar métricas, resultados ou conclusões.
- Salvar resultados em CSV de forma consistente.
- Não alterar hiperparâmetros para “melhorar” resultados sem autorização.

## Git

- Seguir Conventional Commits: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.
- Branches semânticas: `feat/q1-initialization`, `feat/q2-hyperparameters`, `feat/q3-topology`, `feat/q4-training-size`, `feat/q5-feature-selection`, `feat/q6-final-evaluation`, `feat/q7-cross-validation`, `docs/final-report`, `fix/nome-da-correcao`.
- Exemplos:
  - `feat(q1): implementa experimentos de inicialização`
  - `feat(q2): adiciona busca de hiperparâmetros`
  - `fix(q3): corrige geração de topologias`
  - `docs: atualiza relatório técnico`
- Nunca fazer commit, push ou merge sem autorização humana.

## Processo de implementação

Para cada solicitação:
1. Consultar o planejamento.
2. Delimitar o escopo.
3. Inspecionar o código existente.
4. Listar os arquivos/células afetados.
5. Implementar somente o necessário.
6. Executar as verificações pertinentes.
7. Revisar as alterações.
8. Apresentar a entrega no formato abaixo.

Para experimentos longos, executar apenas o bloco solicitado e informar o tempo; não disparar a grade completa sem autorização.

## Formato obrigatório de entrega

Ao concluir, informar:
- Funcionalidade implementada.
- Arquivos/células modificados.
- Decisões técnicas relevantes.
- Verificações executadas.
- Resultado das verificações.
- Pendências ou limitações.
- Sugestão de mensagem de commit.

Se alguma verificação não foi executada, dizer explicitamente.

## Template de prompt de implementação

```text
Feature: <nome curto>
Objetivo: <1–2 frases>
Referência: PROJECT_PLAN.md, seção <X>; notebook <nome>, seção <Y>
Escopo:
- <o que deve ser feito>
Critérios de aceitação:
- <resultado verificável 1>
- <resultado verificável 2>
Restrições:
- <restrições específicas>
Entrega:
- Implementar no notebook indicado.
- Executar <verificação>.
- Responder no formato de entrega do AGENTS.md.
```
