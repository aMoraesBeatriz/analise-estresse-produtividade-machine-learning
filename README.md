# analise-estresse-produtividade-machine-learning
Projeto de Machine Learning supervisionado analisando risco de estresse e produtividade no trabalho usando Regressão Logística e Random Forest. Contém ETL, visualização, treinamento dos modelos, métricas e interpretação detalhada dos resultados. Disciplina de Mineração de Dados.

PI1 – Projeto de Machine Learning Supervisionado
Estresse e Produtividade no Trabalho

Foram utilizados dois modelos supervisionados:

- Regressão Logística
- Random Forest

O projeto inclui: ETL, visualização dos dados, treinamento dos modelos, avaliação e interpretação dos resultados.

1. Descrição do Projeto

O projeto analisa dois cenários:

Problema A – Risco de Estresse

Classificação do risco de estresse com base em:

- Horas de sono
- Horas de tela (celular/computador)
- Consumo diário de água

Modelo utilizado: Regressão Logística

Problema B – Produtividade no Trabalho

Classificação de produtividade com base em:

- Horas trabalhadas
- Pausas realizadas
- Atividade física semanal

Modelo utilizado: Random Forest Classifier

2. Tecnologias Utilizadas

- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

3. Estrutura do Código

O código contém:

3.1. ETL e Preparação dos Dados

- Criação de dados fictícios
- Tratamento e organização
- Separação em treino e teste

3.2. Visualização

- Gráficos exploratórios (scatter e boxplots)
- Análise visual da relação entre variáveis

3.3. Modelos de Machine Learning

Modelo 1 – Regressão Logística:
Previsão do risco de estresse.

Modelo 2 – Random Forest:
Classificação da produtividade e análise de importância das variáveis.

3.4. Métricas e Avaliação

- Acurácia
- Classification Report
- Matriz de Confusão
- Importância das variáveis (Random Forest)

4. Interpretação dos Resultados
Modelo 1 – Regressão Logística

A Regressão Logística classificou o risco de estresse analisando hábitos diários.
A acurácia e as métricas detalhadas indicam a capacidade do modelo em diferenciar funcionários com maior ou menor risco.
Horas de sono, horas de tela e ingestão de água mostraram-se fatores relevantes.

Modelo 2 – Random Forest

O Random Forest apresentou boa performance na classificação de produtividade.
A importância das variáveis indica quais hábitos exercem maior impacto no desempenho:

- Horas trabalhadas
- Pausas
- Atividade física

O modelo também permite identificar padrões que podem auxiliar na tomada de decisões sobre saúde ocupacional.

5. Como Executar

Instale as dependências:

- pip install pandas numpy scikit-learn matplotlib seaborn

Execute o script:

- python pi1_machine_learning.py

6. Arquivos do Projeto

- pi1_machine_learning.py – Código completo do projeto

- README.md – Documentação

7. Licença

Este projeto é apenas para fins educacionais e acadêmicos (Disciplina de Mineração de Dados do 8º Período do Curso de Sistemas de Informação da Universidade do Estado de Minas Gerais).
