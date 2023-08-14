import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import openai
import io

# Sentiment analysis
@st.cache(allow_output_mutation=True)  
def analyze_sentiment(symbol):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Analyze the sentiment for {symbol} stock based on recent news and social media posts.",
            max_tokens=50
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        return None

# Stock data API
@st.cache
def get_stock_data(symbol):
    try:
        api_key = "YOUR_API_KEY" 
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
        response = requests.get(url, params={"apikey": api_key})
        data = response.json()
        df = pd.DataFrame(data['historical'])
        return df
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None

# Inflation web scraping
@st.cache
def get_inflation_data():
    try:
        url = "https://www.bls.gov/cpi/tables/supplemental-files/historical-cpi-u-202301.csv"
        response = requests.get(url)
        df = pd.read_csv(io.StringIO(response.text))
        return df 
    except Exception as e:
        st.error(f"Error fetching inflation data: {str(e)}")
        return None

# GDP web scraping
@st.cache
def scrape_gdp():
    try:
        url = "https://www.bea.gov/news/schedule"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        gdp = soup.find('span', {'class': 'chart-output'}).text
        return float(gdp)
    except Exception as e:
        st.error(f"Error fetching GDP data: {str(e)}")
        return None

# Forecasting 
def forecast_trends(df):
    try:
        model = ARIMA(df['close'], order=(5,1,0)) 
        model_fit = model.fit(disp=0)
        forecast = model_fit.forecast(steps=30)
        return forecast
    except Exception as e:
        st.error(f"Error forecasting trends: {str(e)}")
        return None

# Streamlit app
st.title('Financial Data Explorer')

option = st.sidebar.selectbox('Choose data', ['Stock', 'Inflation', 'GDP'])

if option == 'Stock':
    symbol = st.text_input('Enter stock symbol')
    if symbol:
        df = get_stock_data(symbol)
        sentiment = analyze_sentiment(symbol)
        if df is not None and sentiment is not None:
            st.write('Sentiment:', sentiment)
            st.line_chart(df['close'])
        
elif option == 'Inflation':
    df = get_inflation_data()
    if df is not None:
        forecast = forecast_trends(df)
        st.line_chart(df['CPIAUCSL'])
        st.line_chart(forecast)
    
else:
    gdp = scrape_gdp()
    if gdp is not None:
        st.metric('GDP', gdp)
    
st.header('Ask a finance question')
question = st.text_input('Enter question')
if question:
    try:
        response = openai.Completion.create(
            engine="text-davinci",
            prompt=question,
            max_tokens=100
        )
        st.write(response.choices[0].text)
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

st.caption('Created by gsic')
