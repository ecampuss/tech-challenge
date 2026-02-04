import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import OneHotEncodingNames, MinMax
from sklearn.pipeline import Pipeline
import joblib
from joblib import load

dados = pd.read_csv(
    "https://raw.githubusercontent.com/ecampuss/tech-challenge/refs/heads/main/Obesity.csv"
)


st.set_page_config(page_title="Health Profile Evaluation", layout="centered")

st.title("Health Profile Evaluation")
st.write("Fill the fields below to see the probability of overweight:")

with st.form("form_user"):

    # perguntas categoricas
    Gender = st.selectbox("Gender", ["Female", "Male"])

    family_history = st.selectbox(
        "Overweight Family History?",
        ["yes", "no"]
    )
    scc_dict = {'yes': 1, 'no': 0}

    FAVC = st.selectbox(
        "Frequent consumption of high-caloric food?",
        ["yes", "no"]
    )
    scc_dict = {'yes': 1, 'no': 0}

    CAEC = st.selectbox(
        "Consumption of food between meals?",
        ["no", "Sometimes", "Frequently", "Always"]
    )

    SMOKE = st.selectbox(
        "Smoking?",
        ["yes", "no"]
    )
    scc_dict = {'yes': 1, 'no': 0}

    SCC = st.selectbox(
        "Do you monitor your Calories consumption?",
        ["yes", "no"]
    )
    scc_dict = {'yes': 1, 'no': 0}

    CALC = st.selectbox(
        "Alcohol consumption?",
        ["no", "Sometimes", "Frequently"]
    )

    MTRANS = st.selectbox(
        "What is your Transportation used / Mode of transport?",
        [
            "Automobile",
            "Motorbike",
            "Public_Transportation",
            "Walking"
        ]
    )

    # perguntas numericas
    Age = st.number_input(
        "Age (years)",
        min_value=14,
        max_value=61,
        step=1
    )

    Height = st.number_input(
        "Height (m)",
        min_value=1.40,
        max_value=2.10,
        step=0.01
    )

    Weight = st.number_input(
        "Weight (kg)",
        min_value=30.0,
        max_value=200.0,
        step=0.1
    )

    FCVC = st.slider(
        "How frequent do you eat vegetables?",
        min_value=1,
        max_value=3,
        value=2
    )

    NCP = st.slider(
        "Number of main meals per day?",
        min_value=1,
        max_value=4,
        value=3
    )

    CH2O = st.slider(
        "Daily water consumption? (1: less than 1 L/day, 2: between 1–2 L/day, 3: more than 2 L/day.)",
        min_value=1,
        max_value=3,
        value=2
    )

    FAF = st.slider(
        "Physical activity frequency (0: none, 1: ~1–2×/week, 2: ~3–4×/week, 3: 5-X×/week)",
        min_value=0,
        max_value=3,
        value=1
    )

    TUE = st.slider(
        "Time using electronic devices (0: ~0–2 h/day, 1: ~3–5 h/day, 2: > 5 h/day)",
        min_value=0,
        max_value=2,
        value=1
    )

    submitted = st.form_submit_button("Submit")

if submitted:

    predict_df = pd.DataFrame([{
        "Gender": Gender,
        "Age": Age,
        "Height": Height,
        "Weight": Weight,
        "family_history": family_history,
        "FAVC": FAVC,
        "FCVC": int(round(FCVC)),
        "NCP": int(round(NCP)),
        "CAEC": CAEC,
        "SMOKE": SMOKE,
        "CH2O": int(round(CH2O)),
        "SCC": SCC,
        "FAF": int(round(FAF)),
        "TUE": int(round(TUE)),
        "CALC": CALC,
        "MTRANS": MTRANS,
        "Obesity": 0
    }])

    def data_split(df, test_size):
        treino_df, teste_df = train_test_split(df, test_size=test_size)
        return treino_df.reset_index(drop=True), teste_df.reset_index(drop=True)

    treino_df, teste_df = data_split(dados, 0.2)

    #predict_df = pd.DataFrame([dados_usuario], columns=teste_df.columns)
    teste_novo = pd.concat([teste_df, predict_df], ignore_index=True)

    def pipeline_teste(df):

        pipeline = Pipeline([
            ('MinMaxScaler', MinMax()),
            ('OneHotEncoding', OneHotEncodingNames())
        ])

        df_pipeline = pipeline.fit_transform(df)
        return df_pipeline

    teste_novo = pipeline_teste(teste_novo)

    resultado_pred = teste_novo.drop(['Obesity'], axis=1)

    model = joblib.load('modelo/forest.joblib')

    resultado_pred = resultado_pred.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    final_pred = model.predict(resultado_pred)

    if final_pred[-1] == 0:
        st.success('### Continue assim cuidando da sua saúde para evitar a obesidade!')
        st.balloons()
    else:
        st.error('### Você foi diagnosticado com algum grau de obesidade. Favor entrar em contato com a sua unidade de saúde mais próxima. ')



