'''
A stock portfolio optimizer for n stocks
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import math
import yfinance as yf
from matplotlib import ticker
from scipy import optimize
from _pylief import exception


# first - user gives stock tickers
# second - get monthly stock price history for last 60 months
# third - calculate monthly returns based on Adj. Close
# four - find mean and risk of returns
# five - find covariance of each stonk
# six - calculate optimal weights of each stonk based on Sharpe ratio

#TODO:
# fix error message for invalid ticker
# github profile 
# today's risk free rate
# pull sustainalytics/bloomberg disclosure score 
# streamlit
# convert rf rate from monthly data to daily

def validate_input(ticker):
    valid_input = ticker.strip().upper()
    valid_ticker = yf.Ticker(valid_input)
    valid_ticker_info = valid_ticker.info
    if valid_ticker_info['regularMarketPrice'] is None and not valid_input == "STOP": 
        raise Exception("An invalid ticker was entered")
    return valid_input

# a function that creates a list of valid tickers
def get_tickers():
    keep_asking = True
    ticker_lst = []
    while keep_asking:
        raw_ticker = input(f'Input a ticker, when done type "stop":') # one ticker at a time
        ticker = validate_input(raw_ticker)
        if ticker == "STOP":
            keep_asking = False
        else:
            ticker_lst.append(ticker)
    return ticker_lst

def get_holding_period():
    first_date = input(f'Input a start date YYYY-MM-DD:')
    second_date = input(f'Input a end date YYYY-MM-DD: ')
    dates = (first_date,second_date)
    return dates

# a function that gets stock prices in a given time frame
def get_prices(ticker_lst, start_date, end_date):
    tickers_to_stock_data = {}
    for ticker in ticker_lst:
        x = yf.Ticker(ticker) 
        stock_data = x.history(start = start_date, end = end_date)
        tickers_to_stock_data[ticker]= stock_data
    print(tickers_to_stock_data)
    return tickers_to_stock_data 

# a function that calculates discrete monthly returns within given time frame   
def get_monthly_returns(stock_data):        
    try:
        stock_data_monthly_returns = stock_data['Close'].resample('M').ffill().pct_change()
        return stock_data_monthly_returns
    except:
        raise Exception("An invalid date was entered")

def get_all_stocks_monthly_returns(tickers_to_stock_data):
    ticker_to_monthly_returns = {}# open dct of individual ticker to set of monthly returns of N months
    for ticker in tickers_to_stock_data: 
        stock_data = tickers_to_stock_data[ticker] 
        monthly_returns_data = get_monthly_returns(stock_data) 
        ticker_to_monthly_returns[ticker] = monthly_returns_data
    all_stocks_monthly_returns_df = pd.DataFrame(ticker_to_monthly_returns)
    return all_stocks_monthly_returns_df

def get_rf():
    rf=web.get_data_fred('GS10')
    #print(rf)
    rf=rf['GS10'].iat[-1]
    return rf 

def get_sharpe_ratio(weights, covariance, returns, risk_free):
    weights_shaped = np.reshape(weights,(1,-1))
    #print(weights_shaped.shape)
    #print(covariance.shape)
    portfolio_variance = weights_shaped.dot(covariance.dot(weights_shaped.T))
    #print(portfolio_variance.shape)
    portfolio_risk = math.sqrt(portfolio_variance)
    portfolio_return = np.sum(returns*weights)
    sharpe_ratio = (portfolio_return - risk_free)/portfolio_risk
    return sharpe_ratio

def get_negative_sharpe_ratio(weights, covariance, returns, risk_free):
    return -1 * get_sharpe_ratio(weights, covariance, returns, risk_free)

def con_sum_one(weights):
    return weights.sum()-1

def get_optimal_weights(covariances, means, risk_free, tickers):
    initial_guess = np.full(means.shape, 1.0/means.shape[0])
    # define bounds
    b = (0.01, 1.0)
    bounds = []
    for element in tickers:
        bounds.append(b)
    cons   = ({'type':'eq','fun':con_sum_one})
    optimal_weights = optimize.minimize(get_negative_sharpe_ratio, initial_guess, bounds = bounds, constraints = cons, args=(covariances, means, risk_free))
    return optimal_weights   
    
def format_returns(tickers, means):
    print()
    print(f'Stock average monthly returns:')
    for i, ticker in enumerate(tickers): 
        print(f'{ticker:5}: {means[i]:7.2%}')
    print()
    print()
    print()

def format_risks(tickers, risks):
    print(f'Stock average monthly risks:')
    for i, ticker in enumerate(tickers):    
        print(f'{ticker:5}: {risks[i]:7.2%}')
    print()
    print()
    print()
    
def format_covariances(tickers, covariances):
    print(f'Stock Covariances:') 
    print(f'{covariances}')
    print()
    print()
    print()    

def print_stuff(stuff):
    print(stuff*50)
    
def format_results(tickers, weights):
    print_stuff("*")
    print(f'Your Sharpe ratio-weighted optimized portfolio is:')
    print_stuff("*")
    print_stuff("-") 
    for i, ticker in enumerate(tickers): 
        print(f'{ticker:5}: {weights[i]:7.2%}')
        print_stuff("-")
        
##############################################################################################################

def main():    
    tickers = get_tickers() # get tickers function creates a list of tickers
    holding_period = get_holding_period() # calls get_holding_period function to define the holding period
    start_date = holding_period[0] # defines the starting date as the 0th item in holding period
    end_date = holding_period[1] # defines the end date as the last item in holding period
    
    tickers_to_stock_data = get_prices(tickers, start_date, end_date)   # get prices function maps tickers to stock data
    all_stocks_monthly_returns_df = get_all_stocks_monthly_returns(tickers_to_stock_data) # creates a dataframe for all the stocks' monthly returns
    
    means = all_stocks_monthly_returns_df.mean(axis=0) 
    risks = all_stocks_monthly_returns_df.std(axis=0)
    covariances = all_stocks_monthly_returns_df.cov() 
    risk_free=get_rf()
    
    optimal_weights = get_optimal_weights(covariances, means, risk_free, tickers)

    print(f'Welcome to the stock portfolio optimizer')
    print(f'Using {risk_free}% as the most recent monthly 10yr Treasury Rate')

    format_results(tickers, optimal_weights.x) 
    format_returns(tickers, means)
    format_risks(tickers, risks)

#get_esg("BB")
           
main() 
        
        
                