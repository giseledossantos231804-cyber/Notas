import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import gradio as gr

# ---------------------------------------------------------
# 1. Dados de treinamento
# (adicionei mais linhas para o modelo aprender melhor)
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
y = df['Situacao']  # Series (não DataFrame) - evita warnings do sklearn

X_train, X_teste, y_train, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)

# Acurácia correta: avaliada no conjunto de teste (não em 1 aluno novo)
y_pred_teste = modelo.predict(X_teste)
acuracia = accuracy_score(y_teste, y_pred_teste)
print(f"Acurácia do modelo no conjunto de teste: {acuracia:.2%}")

# ---------------------------------------------------------
# 3. Função de previsão usada pela interface
# ---------------------------------------------------------
EMOJI_SITUACAO = {
    'Aprovado': '🟢',
    'Recuperação': '🟡',
    'Reprovado': '🔴',
}

def prever_situacao(horas, faltas, nota):
    if horas is None or faltas is None or nota is None:
        return "⚠️ Preencha todos os campos."
    if horas < 0 or faltas < 0 or not (0 <= nota <= 10):
        return "⚠️ Verifique os valores: horas/faltas não podem ser negativas e a nota deve estar entre 0 e 10."

    df_novo = pd.DataFrame([[horas, faltas, nota]],
                            columns=['Horas_de_estudo', 'Faltas', 'Nota'])
    resultado = modelo.predict(df_novo)[0]

    # nível de confiança da previsão
    probabilidades = modelo.predict_proba(df_novo)[0]
    prob_dict = dict(zip(modelo.classes_, probabilidades))
    confianca = prob_dict[resultado] * 100

    emoji = EMOJI_SITUACAO.get(resultado, '')
    return f"## {emoji} {resultado}\n\nConfiança do modelo: **{confianca:.1f}%**"

# ---------------------------------------------------------
# 4. Interface Gradio — mais intuitiva
#    (sliders em vez de campos numéricos soltos, emojis,
#     validação, exemplos clicáveis e explicação do resultado)
# ---------------------------------------------------------
with gr.Blocks(title="Previsor de Situação Escolar") as interface:
    gr.Markdown(
        """
        # 🎓 Previsor de Situação Escolar
        Preencha os dados do aluno para prever se ele ficará
        **Aprovado**, em **Recuperação** ou **Reprovado**.
        """
    )
    gr.Markdown(f"*Acurácia do modelo no conjunto de teste: {acuracia:.0%}*")

    with gr.Row():
        with gr.Column():
            horas = gr.Slider(0, 20, value=8, step=1,
                               label="📚 Horas de estudo por semana")
            faltas = gr.Slider(0, 30, value=2, step=1,
                                label="🚪 Número de faltas")
            nota = gr.Slider(0, 10, value=7.0, step=0.5,
                              label="📝 Nota atual")
            botao = gr.Button("Prever situação", variant="primary")

        with gr.Column():
            saida = gr.Markdown(label="Resultado")

    botao.click(fn=prever_situacao, inputs=[horas, faltas, nota], outputs=saida)

    gr.Examples(
        label="Exemplos rápidos (clique para testar)",
        examples=[[10, 1, 9.0], [2, 18, 3.0], [6, 6, 6.5]],
        inputs=[horas, faltas, nota],
    )

if __name__ == "__main__":
    interface.launch()
