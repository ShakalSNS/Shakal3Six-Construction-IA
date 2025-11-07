# app.py - Shakal3Six Construction IA (Saint-Hyacinthe)
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import streamlit as st
import requests

# === 30 LEADS REALES (ft²) ===
data = pd.DataFrame({
    'descripcion': [  # Tus 30 leads reales aquí (copia de v10)
        'Renovacion escuela 5382 ft² Monteregie bajo costo',
        # ... (todas las 30)
    ],
    'presupuesto': [150000, 500000, 80000, 120000, 2000000, 180000, 350000, 120000, 280000, 90000,
                    420000, 110000, 250000, 85000, 320000, 380000, 75000, 550000, 160000, 130000,
                    95000, 220000, 190000, 70000, 60000, 300000, 250000, 70000, 420000, 380000],
    'ubicacion': ['Monteregie','Montreal Rive Sud','Estrie','Monteregie','Montreal Rive Sud'] * 6,
    'etiqueta': ['Ganado','Perdido','Ganado','Ganado','Perdido'] * 6
})

# === MODELO ===
preprocessor = ColumnTransformer([
    ('text', TfidfVectorizer(stop_words='english'), 'descripcion'),
    ('num', MinMaxScaler(), ['presupuesto']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['ubicacion'])
], sparse_threshold=0)

pipeline = Pipeline([('prep', preprocessor), ('clf', MultinomialNB())])
pipeline.fit(data[['descripcion', 'presupuesto', 'ubicacion']], data['etiqueta'])

# === GROK API ===
def generar_oferta(lead):
    try:
        api_key = st.secrets["GROK_API_KEY"]
        prompt = f"Soumission pour {lead['descripcion']}, budget {lead['presupuesto']:,} CAD, lieu {lead['ubicacion']}. Prix, délai, garantie. Max 150 mots."
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            json={"model": "grok-beta", "messages": [{"role": "user", "content": prompt}]},
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return response.json()['choices'][0]['message']['content']
    except:
        return f"**SOUMISSION** – {lead['descripcion']} – Prix: {lead['presupuesto']*1.25:,.0f} CAD – 8 semaines – @Shakal3Six"

# === APP ===
st.title("🤖 Agent Construction – Saint-Hyacinthe")
desc = st.text_input("📋 Projet")
pres = st.number_input("💰 Budget (CAD)", value=300000)
ubi = st.selectbox("📍 Région", ['Monteregie', 'Estrie', 'Montreal Rive Sud'])

if st.button("🚀 Analyser"):
    nuevo = pd.DataFrame([{'descripcion': desc, 'presupuesto': pres, 'ubicacion': ubi}])
    prob = pipeline.predict_proba(nuevo)[0]
    confianza = max(prob)
    decision = 'APTO' if confianza > 0.7 else 'RECHAZAR'
    st.write(f"**Décision**: {decision} | **Confiance**: {confianza:.1%}")
    if decision == 'APTO':
        oferta = generar_oferta(nuevo.iloc[0])
        st.success("**SOUMISSION PRÊTE**")
        st.markdown(oferta)
