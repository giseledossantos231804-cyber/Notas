import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# ---------------------------------------------------------
# 1. Dados de treinamento
# ---------------------------------------------------------
dados = {
    'Horas_de_estudo': [10, 2, 5, 8, 1, 9, 3, 7, 4, 6],
    'Faltas':          [2, 15, 6, 1, 20, 0, 12, 3, 10, 5],
    'Nota':            [8.5, 3.0, 6.5, 9.0, 2.5, 9.5, 4.0, 7.5, 5.0, 6.0],
    'Situacao': ['Aprovado', 'Reprovado', 'Recuperação', 'Aprovado', 'Reprovado',
                 'Aprovado', 'Reprovado', 'Aprovado', 'Recuperação', 'Recuperação']
}

df = pd.DataFrame(dados)

# ---------------------------------------------------------
# 2. Preparação e treino do modelo
# ---------------------------------------------------------
X = df[['Horas_de_estudo', 'Faltas', 'Nota']]
y = df['Situacao']

X_train, X_teste, y_train, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)

# Acurácia avaliada no conjunto de teste
y_pred_teste = modelo.predict(X_teste)
acuracia = accuracy_score(y_teste, y_pred_teste)

# ---------------------------------------------------------
# 3. Configuração da página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Previsor de Situação Escolar",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Previsor de Situação Escolar")
st.write(
    "Preencha os dados do aluno para prever se ele ficará "
    "**Aprovado**, em **Recuperação** ou **Reprovado**."
)

st.info(f"📊 Acurácia do modelo no conjunto de teste: **{acuracia:.0%}**")

# ---------------------------------------------------------
# 4. Entrada de dados
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    horas = st.slider(
        "📚 Horas de estudo por semana",
        min_value=0,
        max_value=20,
        value=8,
        step=1
    )

    faltas = st.slider(
        "🚪 Número de faltas",
        min_value=0,
        max_value=30,
        value=2,
        step=1
    )

with col2:
    nota = st.slider(
        "📝 Nota atual",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.5
    )

# ---------------------------------------------------------
# 5. Previsão
# ---------------------------------------------------------
if st.button("🔮 Prever situação", type="primary", use_container_width=True):

    df_novo = pd.DataFrame(
        [[horas, faltas, nota]],
        columns=['Horas_de_estudo', 'Faltas', 'Nota']
    )

    resultado = modelo.predict(df_novo)[0]

    # Nível de confiança da previsão
    probabilidades = modelo.predict_proba(df_novo)[0]
    prob_dict = dict(zip(modelo.classes_, probabilidades))
    confianca = prob_dict[resultado] * 100

    emoji_situacao = {
        'Aprovado': '🟢',
        'Recuperação': '🟡',
        'Reprovado': '🔴',
    }

    emoji = emoji_situacao.get(resultado, '')

    st.subheader(f"{emoji} {resultado}")
    st.write(f"Confiança do modelo: **{confianca:.1f}%**")

# ---------------------------------------------------------
# 6. Exemplos rápidos
# ---------------------------------------------------------
st.divider()
st.subheader("💡 Exemplos rápidos")

exemplos = {
    "🟢 Aluno com bom desempenho": [10, 1, 9.0],
    "🔴 Aluno com baixo desempenho": [2, 18, 3.0],
    "🟡 Aluno intermediário": [6, 6, 6.5],
}

for nome, valores in exemplos.items():
    with st.expander(nome):
        st.write(f"📚 Horas de estudo: **{valores[0]}**")
        st.write(f"🚪 Faltas: **{valores[1]}**")
        st.write(f"📝 Nota: **{valores[2]}**")
        st.caption("Use esses valores nos controles acima e clique em «Prever situação».")

st.divider()
st.caption(
    "⚠️ Os dados utilizados são dados de exemplo e têm finalidade educacional. "
    "A previsão não deve ser utilizada como avaliação real do desempenho de um aluno."
)
