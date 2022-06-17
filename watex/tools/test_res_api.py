# -*- coding: utf-8 -*-
# Created on Tue May 18 21:05:29 2021
# @author: @Daniel03

# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from plotly.offline import plot
# import matplotlib.pyplot as plt
# import datetime
# from pycoingecko import CoinGeckoAPI
# from mplfinance.original_flavor import candlestick2_ohlc


# def resAPIs(): 
#     """
#     Example to deal with RES (Representation State Transfer) 
    
#     """
#     !pip install pycoingecko 

#     cg =CoinGeckoAPI()
#     bitcoin_data=cg.get_coin_market_chart_by_id(id='bitcoin', vs_currently ='usd', 
#                                                 days=30)
    
#     # we are interessing in preices 
#     bitcoin_data['prices']
#     #convert data to pandas 
#     data =pd.DataFrame(bitcoin_price_data, columns=['TimeStamp', 'Prices'])
#     #convert padans TimeStamp to datetime 
#     data['Date'] =pd.to_datetime(data['TimeStamp'], unit='ms')
#     #create candlestick_data 
#     candlestick_data =data.groupby(data.Date.dt.date).agg({'Price':['min',
#                                                                     'max',
#                                                                     'first', 
#                                                                     'last']})
#     fig=go.Figure(data=[go.Candlestick(x=candlestick_data.index, 
#                     open=candlestick_data['Price']['first'], 
#                     high =candlestick_data['Price']['max'], 
#                     low=candlestick_data['Price']['min'], 
#                     close =candlestick_data['Price']['last'])
#                     ])
#     fig.update_layout(xaxis_rangeslider_visible=False, xaxis_title ='Date', 
#                       yaxis_title ='Price (USD $)', 
#                       title='Bitcoin Candlestick chart Over Past 30 Days')
    
#     plot(fig, filename='bitcoin_candlesick_graph.html')
    
    ###########################################
    
    