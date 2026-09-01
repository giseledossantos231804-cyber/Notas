
# 🎓 Previsor de Situação Escolar

Aplicação de Machine Learning desenvolvida em Python para prever a situação escolar de um aluno com base em:

* 📚 Horas de estudo por semana
* 🚪 Número de faltas
* 📝 Nota atual

O modelo utilizado é uma **Árvore de Decisão (`DecisionTreeClassifier`)**.

## 🤖 Como funciona

O modelo recebe três informações:

| Informação      | Descrição                                |
| --------------- | ---------------------------------------- |
| Horas de estudo | Quantidade de horas estudadas por semana |
| Faltas          | Número de faltas do aluno                |
| Nota            | Nota atual do aluno                      |

A partir desses dados, o modelo classifica o aluno em uma das três situações:

🟢 **Aprovado**

🟡 **Recuperação**

🔴 **Reprovado**

Além da situação prevista, a aplicação apresenta a **confiança do modelo** para aquela previsão.

## 📊 Dados utilizados

O modelo foi treinado com um conjunto de dados contendo informações sobre horas de estudo, faltas, notas e situação escolar.

Os dados estão definidos diretamente no arquivo `index.py`.

## 🧠 Modelo de Machine Learning

Foi utilizada uma **Árvore de Decisão**, implementada através da biblioteca Scikit-learn.

O conjunto de dados é dividido em:

* 80% para treinamento
* 20% para teste

A acurácia é calculada utilizando o conjunto de teste.

## 🖥️ Interface

A interface da aplicação foi desenvolvida utilizando **Gradio**, permitindo que o usuário informe os dados através de controles deslizantes.

## 🚀 Tecnologias utilizadas

* Python
* Pandas
* Scikit-learn
* Gradio
* Machine Learning
* Árvore de Decisão

## 📁 Estrutura do projeto

```text
.
├── index.py
├── requirements.txt
└── README.md
```

## ▶️ Como executar

Instale as dependências:

```bash
pip install -r requirements.txt
```

Depois execute:

```bash
python index.py
```

A aplicação será iniciada através do Gradio.

## ⚠️ Observação

Este projeto possui finalidade educacional e demonstra o funcionamento de um modelo simples de Machine Learning para classificação de situação escolar.
