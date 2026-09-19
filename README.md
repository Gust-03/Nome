# Projeto de Machine Learning — Classificação de Vinhos
Sobre o projeto
Este projeto apresenta uma aplicação de Machine Learning supervisionado utilizando o dataset Wine, disponibilizado pela biblioteca scikit-learn.
O objetivo é analisar características de diferentes tipos de vinho de um set de dados e desenvolver um modelo capaz de classificar automaticamente as amostras em três classes distintas.
Durante o projeto são realizadas etapas de:
Carregamento e preparação dos dados;
Análise exploratória do dataset;
Verificação de valores ausentes e duplicados;
Análise estatística das variáveis;
Visualização dos dados;
Análise de correlação entre características;
Separação entre variáveis preditoras e variável-alvo;
Divisão dos dados em treinamento e teste;
Treinamento de uma Árvore de Decisão;
Avaliação do modelo;
Geração de matriz de confusão;
Análise de classificação;
Visualização da árvore de decisão;
Investigação da relação entre complexidade do modelo e desempenho, com foco em overfitting.

O dataset utilizado é o Wine Dataset, carregado diretamente através do scikit-learn:

from sklearn.datasets import load_wine
vino = load_wine()


Os dados são posteriormente transformados em um DataFrame do Pandas:

qd = pd.DataFrame(
    vino.data,
    columns=vino.feature_names
qd["target"] = vino.target


Dessa forma, as características químicas são armazenadas como colunas e a classe correspondente é adicionada na coluna target.

Análise exploratória dos dados

Antes do treinamento do modelo, é realizada uma análise inicial para compreender a estrutura do dataset.
São verificados:
Primeiras linhas do dataset;
Dimensões;
Tipos de dados;
Valores ausentes;
Valores duplicados;
Estatísticas descritivas;
Distribuição das classes.

Exemplo:

print("Dimensões:", qd.shape)
display(qd.dtypes)
display(qd.isna().sum())
print(qd.duplicated().sum())
display(qd.describe())


Essa etapa permite identificar possíveis problemas nos dados antes da aplicação do algoritmo de Machine Learning.

Visualização dos dados
O projeto utiliza Matplotlib e Seaborn para visualizar diferentes características do dataset.

Distribuição das classes
É utilizado um gráfico de contagem para observar a quantidade de amostras pertencentes a cada classe:

sns.countplot(
    data=qd,
    x="target"
)


Essa visualização permite verificar como as amostras estão distribuídas entre as classes.

Relação entre álcool e intensidade da cor

Também é criado um gráfico de dispersão relacionando os parâmetros: "alcohol" e "color_intensity"
sns.scatterplot(
    data=qd,
    x="alcohol",
    y="color_intensity",
    hue="target"
)


Essa visualização ajuda a identificar possíveis padrões ou separações entre as classes.

Matriz de correlação
A correlação entre as variáveis também é analisada:
sns.heatmap(
    qd.drop(columns="target").corr(),
    cmap="coolwarm"
)

A matriz de correlação permite observar quais características possuem maior ou menor relação linear entre si.

Preparação dos dados
Após a análise exploratória, os dados são divididos em:

X: variáveis utilizadas para realizar as previsões;
y: variável-alvo que representa a classe do vinho.
X = qd.drop(columns="target")
y = qd["target"]

Em seguida, o dataset é dividido em conjuntos de treinamento e teste:
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


Foi utilizado:
80% dos dados para treinamento
20% dos dados para teste
random_state=42 para permitir a reprodução dos resultados.

Modelo — Árvore de Decisão
O principal algoritmo utilizado no projeto é o Decision Tree Classifier.
A árvore é configurada com profundidade máxima igual a 3:

arv_dcs = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

Depois, o modelo é treinado utilizando os dados de treinamento:

arv_dcs.fit(X_train, y_train)

E utilizado para realizar previsões:
y_pred = arv_dcs.predict(X_test)

Avaliação do modelo
Para avaliar o desempenho da árvore de decisão, são utilizadas diferentes métricas.

Acurácia

A acurácia representa a proporção de previsões realizadas corretamente:

accuracy_score(y_test, y_pred)


O resultado é apresentado tanto em formato decimal quanto no relatório de classificação.

Relatório de classificação
O projeto também utiliza:
classification_report(
    y_test,
    y_pred,
    target_names=vino.target_names
)


Esse relatório apresenta métricas como:
Precision: precisão das previsões de cada classe;
Recall: capacidade de identificar corretamente as amostras de cada classe;
F1-score: combinação entre precision e recall;
Support: quantidade de amostras pertencentes a cada classe.

Matriz de confusão
A matriz de confusão permite visualizar a quantidade de classificações corretas e incorretas para cada classe.

mc = confusion_matrix(y_test, y_pred)

O resultado é apresentado utilizando um heatmap:
sns.heatmap(
    mc,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=vino.target_names,
    yticklabels=vino.target_names
)


A matriz facilita a identificação de quais classes foram confundidas pelo modelo.

Visualização da árvore
O projeto também apresenta graficamente a estrutura criada pelo algoritmo:

plot_tree(
    arv_dcs,
    feature_names=X.columns,
    class_names=vino.target_names,
    filled=True,
    rounded=True,
    fontsize=9
)


Essa representação permite observar:
Quais características são utilizadas nas decisões;
Os pontos de divisão dos dados;
A profundidade da árvore;
As classes presentes em cada nó.
Uma das principais vantagens da Árvore de Decisão é justamente sua capacidade de apresentar as regras utilizadas para realizar as classificações de maneira visual.

Análise de Overfitting
Uma das etapas finais do projeto consiste em analisar como a profundidade da árvore influencia o desempenho do modelo.
São testadas árvores com profundidades de 1 até 10:

profundidades = range(1, 11)

Para cada profundidade, são calculadas as acurácias de treinamento e teste:
treino.append(arv_dcs.score(X_train, y_train))
teste.append(arv_dcs.score(X_test, y_test))

Os resultados são posteriormente apresentados em um gráfico comparando:
Acurácia de treinamento;
Acurácia de teste;
Profundidade da árvore.
Essa análise permite observar o comportamento do modelo conforme sua complexidade aumenta.

Fluxo do projeto

O funcionamento geral do projeto pode ser resumido da seguinte forma:

Dataset Wine
     ↓
Carregamento dos dados
     ↓
Conversão para DataFrame
     ↓
Análise exploratória
     ↓
Visualização dos dados
     ↓
Análise de correlação
     ↓
Separação X e y
     ↓
Treino / Teste
     ↓
Árvore de Decisão
     ↓
Previsões
     ↓
Avaliação do modelo
     ↓
Matriz de Confusão
     ↓
Visualização da árvore
     ↓
Análise de Overfitting

Conclusão
O projeto demonstra, de forma prática, o processo de construção de um modelo de Machine Learning para classificação de vinhos, desde a análise inicial dos dados até a avaliação do modelo.

A utilização de gráficos, métricas de classificação, matriz de confusão e análise de profundidade da árvore permite compreender não apenas o resultado das previsões, mas também alguns aspectos relacionados ao comportamento e à complexidade do modelo.

O projeto serve como uma introdução prática ao fluxo de desenvolvimento de modelos de classificação utilizando Python e Scikit-learn.
