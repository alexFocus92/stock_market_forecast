# -*- coding: utf-8 -*-

import yfinance as yf
import streamlit as st
import pandas as pd
from prophet import Prophet

def update_tickers():
    df_list_0 = pd.read_html('https://topforeignstocks.com/foreign-adrs-list/the-full-list-of-spanish-adrs/')[0]
    df_list_1 = pd.read_html('https://topforeignstocks.com/foreign-adrs-list/the-full-list-of-spanish-adrs/')[1]
    df_list = df_list_0.append(df_list_1)
    df_list.to_csv('spain_ticker_list.csv', index = False)
    print("The CSV File have updated ok")


def main():
    st.sidebar.title("Forecasting IBEX 35")

    # Convert the file to an opencv image.
    st.image("índice.jpg", channels="BGR", width=700)

    df_list = pd.read_csv('spain_ticker_list.csv')

    # Creacion de los botones de la barra lateral
    company_name = st.sidebar.selectbox("Chose company", df_list['Company Name'])
    changepoint_range = st.sidebar.number_input('changepoint_range',
                                                 min_value=0.1, max_value=0.99,
                                                 value=0.5, step=0.1)
    changepoint_prior_scale = st.sidebar.number_input('changepoint_prior_scale',
                                             min_value=0.5, max_value=5.0,
                                             value=1.0, step=0.1)

    st.write(company_name)
    company_ticker = df_list.loc[df_list['Company Name'] == company_name].Ticker.reset_index(drop=True)[0]
    st.write(company_ticker)

    if st.sidebar.button("Predict", key="predict"):
        #define the ticker symbol
        tickerSymbol = company_ticker
        st.write("Wait 1 minute for the Results")
        st.write("Making predictions...")

        #get data on this ticker
        tickerData = yf.Ticker(tickerSymbol)

        #get the historical prices for this ticker
        tickerDf = tickerData.history(period='1d', start='2019-1-1', end='2023-3-1')

        df = tickerDf['Close']

        df = df.reset_index()
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
        df['ds'] = df['ds'].dt.tz_localize(None)

        m = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                    changepoint_range=changepoint_range)

        m.add_country_holidays(country_name='ES')

        m.fit(df)  # df is a pandas.DataFrame with 'y' and 'ds' columns
        future = m.make_future_dataframe(periods=180)
        predictions = m.predict(future)
        predictions = predictions[predictions['ds'].dt.dayofweek < 5]
        df_pred = predictions[['ds','trend','yhat','yhat_lower','yhat_upper']]
        df_pred['y']=df['y']
        df_pred = df_pred.set_index('ds')

        st.subheader("Prediction")
        st.line_chart(df_pred)

        print(predictions['weekly'])

        fig_components = m.plot_components(predictions)
        st.pyplot(fig_components)

        #see your data
        st.dataframe(tickerDf)


if __name__ == '__main__':
    main()
















