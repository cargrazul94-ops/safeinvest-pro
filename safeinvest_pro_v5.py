import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from transformers import pipeline

st.set_page_config(page_title="SafeInvest Pro v5", layout="wide")
st.title("🚀 SafeInvest Pro v5 - FinBERT + Sentimiento Histórico")
st.markdown("**Análisis Fundamental + Técnico + FinBERT Sentiment (Noticias)**")

# Cargar FinBERT (solo una vez)
@st.cache_resource
def load_finbert():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)  # CPU

finbert = load_finbert()

# Sidebar
st.sidebar.header("Configuración")
strategy = st.sidebar.selectbox("Estrategia", ["Combinada Recomendada", "Value + Calidad", "Momentum Seguro", "Defensivo"])
sentiment_threshold = st.sidebar.slider("Umbral Sentimiento FinBERT", -1.0, 1.0, 0.1, 0.05)

tickers = ["AAPL","MSFT","GOOGL","AMZN","NVDA","JPM","V","JNJ","PG","XOM","KO",
           "SPY","VOO","QQQ","SCHD","VYM","DGRO","VTI","BND","GLD"]

@st.cache_data(ttl=900)
def get_full_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="1y")
    news = stock.news[:10]  # Últimas noticias
    return info, hist, news

def get_finbert_sentiment(text):
    if not text:
        return 0.0
    result = finbert(text[:512])[0]  # Limitar longitud
    score = result['score'] if result['label'] == 'positive' else -result['score']
    return score

data = []
for tick in tickers:
    try:
        info, hist, news_list = get_full_data(tick)
        if hist.empty:
            continue
            
        current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        
        # Momentum y técnicos
        mom_6m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-126] - 1) * 100 if len(hist) > 126 else 0
        rsi = 100 - (100 / (1 + (hist['Close'].diff(1).clip(lower=0).ewm(span=14).mean() / 
                             abs(hist['Close'].diff(1).clip(upper=0)).ewm(span=14).mean())))
        rsi = rsi.iloc[-1]
        sma50 = hist['Close'].rolling(50).mean().iloc[-1]
        
        # Sentimiento FinBERT + Histórico
        sentiments = []
        news_details = []
        for item in news_list:
            title = item.get('title', '')
            summary = item.get('summary', '') or ''
            text = title + ". " + summary
            score = get_finbert_sentiment(text)
            sentiments.append(score)
            news_details.append({"title": title, "score": round(score, 3), "link": item.get('link', '')})
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        sentiment_trend = np.mean(sentiments[-5:]) - np.mean(sentiments[:5]) if len(sentiments) > 8 else 0  # Tendencia reciente
        
        # Señal final
        signal = "MANTENER"
        reasons = []
        
        if mom_6m > 8 and rsi < 70 and avg_sentiment > sentiment_threshold and current_price > sma50 * 0.98:
            signal = "COMPRAR"
            reasons.append(f"FinBERT Positivo ({avg_sentiment:.2f}) + Momentum")
        elif avg_sentiment < -0.3 or rsi > 75 or sentiment_trend < -0.4:
            signal = "VENDER"
            reasons.append(f"FinBERT Negativo o caída fuerte en sentimiento")
        elif abs(sentiment_trend) > 0.5:
            reasons.append(f"Cambio drástico en sentimiento: {sentiment_trend:+.2f}")
        
        row = {
            "Ticker": tick,
            "Precio": round(current_price, 2),
            "Momentum 6M (%)": round(mom_6m, 1),
            "RSI": round(rsi, 1),
            "Sentimiento FinBERT": round(avg_sentiment, 3),
            "Tendencia Sent.": round(sentiment_trend, 3),
            "Señal": signal,
            "Justificación": " | ".join(reasons) if reasons else "Neutral"
        }
        data.append(row)
    except:
        continue

df = pd.DataFrame(data)

st.header("📈 Resumen del Mercado")
spy = yf.Ticker("SPY").info.get("currentPrice", 0)
st.metric("S&P 500 (SPY)", f"${spy:.2f}")

col1, col2 = st.columns(2)
with col1:
    st.subheader("🟢 Para COMPRAR")
    st.dataframe(df[df["Señal"] == "COMPRAR"], use_container_width=True)
with col2:
    st.subheader("🔴 Para VENDER")
    st.dataframe(df[df["Señal"] == "VENDER"], use_container_width=True)

st.header("📊 Análisis Completo")
st.dataframe(df.sort_values("Sentimiento FinBERT", ascending=False), use_container_width=True, height=600)

# Análisis Detallado
st.header("🔍 Detalle + Evolución de Sentimiento")
selected = st.selectbox("Selecciona Ticker", df["Ticker"])
if selected:
    info, hist, news_list = get_full_data(selected)
    row = df[df["Ticker"] == selected].iloc[0]
    
    st.subheader(f"{selected} → **{row['Señal']}**")
    st.write("**Justificación:**", row["Justificación"])
    
    # Gráfico Sentimiento Histórico (simulado por noticias recientes)
    if news_list:
        sent_scores = [get_finbert_sentiment(n.get('title','') + ". " + n.get('summary','')) for n in news_list]
        dates = pd.date_range(end=datetime.now(), periods=len(sent_scores))
        sent_df = pd.DataFrame({"Fecha": dates, "Sentimiento": sent_scores})
        
        fig_sent = px.line(sent_df, x="Fecha", y="Sentimiento", title="Evolución Sentimiento FinBERT (Noticias Recientes)")
        fig_sent.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig_sent, use_container_width=True)
    
    # Noticias
    st.subheader("📰 Noticias Recientes")
    for n in news_list[:5]:
        st.markdown(f"**{n.get('title')}**  \n{n.get('summary','')[:200]}...")
        st.caption(f"Link: {n.get('link','')}")

st.success("**FinBERT** es mucho más preciso que TextBlob en contexto financiero. Monitorea la **Tendencia Sent.** para detectar cambios rápidos.")
st.caption("⚠️ App educativa • No es asesoramiento financiero • DYOR • Datos con posible delay.")
