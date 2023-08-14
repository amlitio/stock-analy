import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import openai
import io

# Set OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

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
        return None

# Stock data API
@st.cache_data
def get_stock_data(symbol):
    try:
        api_key = st.secrets["FINANCIAL_MODELING_PREP_API_KEY"]
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
        response = requests.get(url, params={"apikey": api_key})
        data = response.json()
        df = pd.DataFrame(data['historical'])
        return df
    except Exception as e:
        return None

# Inflation data
@st.cache_data
def get_inflation_data():
    try:
        url = "https://www.bls.gov/cpi/tables/supplemental-files/historical-cpi-u-202301.csv"
        response = requests.get(url)
        df = pd.read_csv(io.StringIO(response.text))
        return df
    except Exception as e:
        return None

# GDP data
@st.cache_data
def scrape_gdp():
    try:
        #url = "https://www.bea.gov/news/schedule"
        url = "https://www.bea.gov/data/gdp/gross-domestic-product"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        gdp = soup.find('span', {'class': 'chart-output'}).text
        return float(gdp)
    except Exception as e:
        return None

# Forecasting
def forecast_trends(df):
    try:
        model = ARIMA(df['close'], order=(5,1,0))
        model_fit = model.fit(disp=0)
        forecast = model_fit.forecast(steps=30)
        return forecast
    except Exception as e:
        return None

# Streamlit app
st.title('Financial Data Explorer')
option = st.sidebar.selectbox('Choose data', ['Stock', 'Inflation', 'GDP'])

if option == 'Stock':
    symbol = st.text_input('Enter stock symbol')

    if symbol:
        df = get_stock_data(symbol)

        if df is None:
            st.error("Error fetching stock data")

        else:
            sentiment = analyze_sentiment(symbol)

            if sentiment is None:
                st.error("Error analyzing sentiment")

            else:
                st.write(sentiment)
                st.line_chart(df['close'])

elif option == 'Inflation':
    df = get_inflation_data()

    if df is None:
        st.error("Error fetching inflation data")

    else:
        forecast = forecast_trends(df)
        st.line_chart(df['CPIAUCSL'])
        st.line_chart(forecast)

else:
    gdp = scrape_gdp()

    if gdp is None:
        st.error("Error fetching GDP data")

    else:
        st.metric('GDP', gdp)

# Streamlit front
st.header('Ask a finance question')
question = st.text_input('Enter question')
if question:
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=question,
            max_tokens=100
        )
        st.write(response.choices[0].text)
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

st.caption('Created by gsic')
