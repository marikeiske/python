from flask import Flask, render_template
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


base = pd.read_csv(r"C:\Users\Estudante\Desktop\app\socialmedia.csv")

#Separar as variáveis independentes (x) e dependente (y)
x = base[["Shares/Retweets"]]
y = base["Likes/Reactions"]

#Dividir os dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Criar e treinar o modelo de regressão linear
reg = LinearRegression()
reg.fit(x_train, y_train)

#Avaliar o modelo nos dados de treinamento
r2_train = reg.score(x_train, y_train)
#print(f"R² Score (Treinamento): {r2_train}")

#Avaliar o modelo nos dados de teste
r2_test = reg.score(x_test, y_test)
#print(f"R² Score (Teste): {r2_test}")

#Criar o gráfico de dispersão com a regressão
fig = px.scatter(base, x="Shares/Retweets", y="Likes/Reactions", trendline='ols', template="plotly_dark")
fig.show()

app = Flask(__name__)

@app.route("/")
def home():
    x = base[["Shares/Retweets"]]
    y = base["Likes/Reactions"]
    fig = px.scatter(base, x="Shares/Retweets", y="Likes/Reactions", trendline='ols', template="plotly_dark")
    return render_template('index.html',plot= fig.to_html())
   
if __name__ == "__main__":
    app.run(debug=True)