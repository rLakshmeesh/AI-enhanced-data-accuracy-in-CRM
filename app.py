import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

st.title("Customer Segmentation, Churn Analysis, and Sentiment Analysis")

st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose a task", ["Customer Segmentation", "Churn Analysis", "Sentiment Analysis"])

@st.cache_data
def load_data():
    return pd.read_csv('cell2cellholdout.csv')

df = load_data()

if option == "Customer Segmentation":
    st.header("Customer Segmentation")
    st.subheader("Dataset")
    st.write(df.head())

    st.subheader("MonthsInService Distribution")
    plot_data = [go.Histogram(x=df['MonthsInService'])]
    fig = go.Figure(data=plot_data, layout=go.Layout(title='MonthsInService'))
    st.plotly_chart(fig)

    st.subheader("Elbow Method for Optimal Clusters")
    sse = {}
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, max_iter=1000, random_state=42).fit(df[['CustomerID', 'MonthsInService']])
        df["MonthsInServiceCluster"] = kmeans.labels_
        sse[k] = kmeans.inertia_

    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
    st.pyplot(plt)

    st.subheader("KMeans Clustering")
    k = st.slider("Select number of clusters", 1, 10, 3)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df[['MonthsInService']])
    df['MonthsInServiceCluster'] = kmeans.predict(df[['MonthsInService']])
    st.write(df[['CustomerID', 'MonthsInService', 'MonthsInServiceCluster']].head())

elif option == "Churn Analysis":
    st.header("Churn Analysis")
    df_churn = df.drop(['HandsetPrice', 'HandsetModels', 'Homeownership', 'MaritalStatus', 'UniqueSubs',
                        'PeakCallsInOut', 'OffPeakCallsInOut', 'DroppedBlockedCalls', 'RetentionCalls',
                        'InboundCalls', 'OverageMinutes', 'ReceivedCalls', 'OwnsMotorcycle', 'NonUSTravel',
                        'OwnsComputer', 'RVOwner', 'TruckOwner', 'HandsetRefurbished', 'HandsetWebCapable',
                        'Handsets'], axis=1)

    binary_cols = []
    multi_Value = []
    for col in df_churn.columns:
        if df_churn[col].dtype == 'object':
            if df_churn[col].nunique() == 2:
                binary_cols.append(col)
            else:
                multi_Value.append(col)

    LE_cat = LabelEncoder()
    for col in multi_Value:
        df_churn[col] = LE_cat.fit_transform(df_churn[col].astype(str))

    Binary_cols_except_churn = binary_cols[1:]
    dfDummies = pd.get_dummies(df_churn[Binary_cols_except_churn], prefix=Binary_cols_except_churn)
    clean_dataframe = pd.concat([df_churn.drop(binary_cols, axis=1), dfDummies, df_churn['Churn']], axis=1)
    clean_dataframe['Churn'].replace({'Yes': 1, 'No': 0}, inplace=True)

    imputer = KNNImputer(n_neighbors=5)
    numeric_columns = clean_dataframe.select_dtypes(include=[np.number]).columns
    clean_dataframe[numeric_columns] = imputer.fit_transform(clean_dataframe[numeric_columns])
    final_dataset = clean_dataframe.reset_index(drop=True)

    y = final_dataset["Churn"]
    X = final_dataset.drop(["Churn", 'CustomerID'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    st.subheader("Model Training and Evaluation")
    model = XGBClassifier().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

elif option == "Sentiment Analysis":
    st.header("Sentiment Analysis")

    @st.cache_data
    def load_sentiment_data():
        return pd.read_csv('tweetsdataset.csv')

    df_sentiment = load_sentiment_data()
    st.subheader("Sentiment Analysis Dataset")
    st.write(df_sentiment.head())

    df_sentiment = df_sentiment.drop(['user_name', 'user_location', 'user_description', 'user_verified', 'date', 'hashtags', 'source'], axis=1)
    df_sentiment = df_sentiment[df_sentiment['label'] != 'neutral'].replace(np.nan, '', regex=True)

    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    df_sentiment['text'] = df_sentiment['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    df_sentiment['text'] = df_sentiment['text'].str.replace('[^\w\s]', '')
    df_sentiment['text'] = df_sentiment['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    import spacy
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    def space(comment):
        return " ".join([token.lemma_ for token in nlp(comment)])
    df_sentiment['text'] = df_sentiment['text'].apply(space)

    st.subheader("Model Training and Evaluation")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB

    X_train, X_test, y_train, y_test = train_test_split(df_sentiment['text'], df_sentiment['label'], test_size=0.25, random_state=5)
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
    train_vectors = vectorizer.fit_transform(X_train)
    test_vectors = vectorizer.transform(X_test)

    model = MultinomialNB().fit(train_vectors, y_train)
    predictions = model.predict(test_vectors)
    st.write("Classification Report:")
    st.text(classification_report(y_test, predictions))
