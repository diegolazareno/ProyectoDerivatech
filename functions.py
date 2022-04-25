"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Derivatech.                                                                                -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: diegolazareno                                                                               -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/diegolazareno/ProyectoDerivatech                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
# Librerías necesarias
import pandas as pd
import numpy as np
import yfinance as yf
import random
import ta
import datetime as dt
import scipy.stats as st
import ipywidgets as widgets
from ipywidgets import interact, interact_manual, interactive

# Conexión a la API de Financial Modeling Prep
from pyfmpcloud import settings
from pyfmpcloud import company_valuation as cv
api_key = pd.read_csv("files/apiKey_FMP.csv").iloc[0, 0]
settings.set_apikey(api_key)

import warnings
warnings.filterwarnings("ignore")

try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen

import certifi
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_jsonparsed_data(url):
    """
    Receive the content of ``url``, parse it as JSON and return the object.

    Parameters
    ----------
    url : str

    Returns
    -------
    dict
    """
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

def userFunction(Company):
    infoNasdaq = pd.read_csv("files/nasdaq_screener_1650511730482.csv")

    # Descarga de información (precios históricos)
    ticker = infoNasdaq[infoNasdaq["Name"] == Company]["Symbol"].values[0]
    ticker = ticker.replace(".", "-")
    end = dt.date.today()
    start = end - dt.timedelta(365 * 5)
    stockPrices = yf.download(ticker, start = start, end = end, progress = False)["Adj Close"]
    rsiStock = ta.momentum.rsi(stockPrices, window = 21)

    # Visual
    fig = plt.figure(figsize = (13, 15), constrained_layout = True) # 13, 15
    spec = fig.add_gridspec(5, 3)
    fig.tight_layout(pad = 4)
    fig.suptitle("Análisis Técnico & Fundamental: " + Company)
            
    # Históricos (precio de cierre ajustado)
    ax0 = fig.add_subplot(spec[0, :])
    #color= ["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)])]
    colors = [("deepskyblue", "royalblue"), ("hotpink", "crimson"), ("steelblue", "navy"), ("lightcoral", "chocolate"),
              ("lightgreen", "seagreen"), ("pink", "purple"), ("tan", "sienna"), ("silver", "grey"), ("salmon", "darkred"),
              ("springgreen", "darkgreen"), ("orchid", "indigo"), ("gold", "goldenrod"), ("orange", "maroon")]
    color = random.choice(colors) 
    ax0.plot(stockPrices, color = color[0], label = ticker)
    ax0.title.set_text("Precio de Cierre Ajustado")
    
    # Señales de compra/venta y estrategia de trading
    long, short = True, True
    backtest = pd.DataFrame(columns = ["Capital"], index = stockPrices.index)
    
    for i in range(len(stockPrices)):
        if rsiStock[i] < 30 and long:
            ax0.plot(stockPrices.index[i], stockPrices[i], marker = 10, color = "green", ms = 9)
            long = False
            short = True
            backtest.iloc[i, 0] = stockPrices[i]
            
        elif rsiStock[i] > 70 and short:
            ax0.plot(stockPrices.index[i], stockPrices[i], marker = 11, color = "red", ms = 9)
            short = False
            long = True
            backtest.iloc[i, 0] = -stockPrices[i]
            
    ax0.legend(loc = "best")
    ax0.set_ylabel(r"\$ (USD)")
    
    # RSI
    ax1 = fig.add_subplot(spec[1, :])
    ax1.plot(rsiStock, color = color[1], label = "RSI 21 periodos")
    ax1.legend(loc = "best")
    ax1.axhline(70, linestyle = "--", alpha = 1, color = "k")
    ax1.axhline(30, linestyle = "--", alpha = 1, color = "k")
    ax1.title.set_text("Análisis Técnico")
    ax1.set_ylabel("RSI")
    ax1.legend(loc = "best")
    
    # Resultados estrategia de trading
    backtest = backtest.dropna()
    if backtest["Capital"][-1] < 0:
        last = stockPrices[-1]
    else:
        last = -stockPrices[-1]
    backtest.loc[stockPrices.index[-1], "Capital"] = last 
    backtest["Profit"] = 0
    for i in range(len(backtest)):
        if i > 0:
            if backtest.iloc[i, 0] < 0:
                backtest.iloc[i, 1] = abs(backtest.iloc[i, 0]) - backtest.iloc[i - 1, 0]
            else:
                backtest.iloc[i, 1] = abs(backtest.iloc[i - 1, 0]) - backtest.iloc[i, 0]
            
    backtest["Profit Acumulado"] = backtest["Profit"].cumsum()
    ax11 = fig.add_subplot(spec[2, :])
    ax11.plot(backtest["Profit Acumulado"], color = color[0], label = "Backtest")
    ax11.title.set_text("Estrategia de Trading")
    ax11.set_ylabel(r"\$ (USD)")
    ax11.legend(loc = "best")
    ax11.set_xlabel("Fecha")
    
    # Análisis fundamental
    url1 = ("https://financialmodelingprep.com/api/v3/income-statement/" + ticker +  "?apikey=a816c18988b47cbfdc9491598c678edd")
    url2 = ("https://financialmodelingprep.com/api/v3/balance-sheet-statement/" + ticker + "?limit=120&apikey=a816c18988b47cbfdc9491598c678edd")
    url3 = ("https://financialmodelingprep.com/api/v3/cash-flow-statement/" + ticker +  "?limit=120&apikey=a816c18988b47cbfdc9491598c678edd")
    
    incomeStat = pd.DataFrame(get_jsonparsed_data(url1))
    balanceSheet = pd.DataFrame(get_jsonparsed_data(url2))
    cashStat = pd.DataFrame(get_jsonparsed_data(url3))
    ratios = cv.financial_ratios(ticker)
    
    intrinsicValuation = pd.DataFrame(columns = ["Net Current Asset Value", "Book Value", "Tangible Book Value", 
                                             "Earnings Power Value"], index = [ticker])
    # Net-Net
    intrinsicValuation.iloc[0, 0] = (balanceSheet["totalCurrentAssets"] - balanceSheet["totalLiabilities"])[0] / incomeStat["weightedAverageShsOut"][0]
    intrinsicValuation.iloc[0, 1] = (balanceSheet["totalAssets"] - balanceSheet["totalLiabilities"])[0] / incomeStat["weightedAverageShsOut"][0]
    intrinsicValuation.iloc[0, 2] = (balanceSheet["totalAssets"] - balanceSheet["totalLiabilities"] - balanceSheet["intangibleAssets"])[0] / incomeStat["weightedAverageShsOut"][0]

    # Earnings Power Value
    roe = np.median(incomeStat["netIncome"].iloc[0 : 5] / balanceSheet["totalStockholdersEquity"].iloc[0 : 5])
    payout = np.median(np.abs(cashStat["dividendsPaid"].iloc[0 : 5]) / incomeStat["netIncome"].iloc[0 : 5])
    g = roe * (1 - payout)
    per = np.median(ratios["priceEarningsRatio"].iloc[0 : 10])

    intrinsicValuation.iloc[0, 3] = incomeStat["eps"][0] * (1 + g) ** 5 * per / (1 + 0.15) ** 5
    
    # Visual 1 (Análisis Fundamental)
    ax2 = fig.add_subplot(spec[3, :])
    ax2.barh(list(intrinsicValuation.columns), list(intrinsicValuation.values[0]), color = color[1], alpha = 0.5)
    ax2.axvline(stockPrices[-1], label = "Último precio: $" + str(round(stockPrices[-1], 2)), 
                linestyle = "--", alpha = 1, color = "k")
    for i in range(len(intrinsicValuation.iloc[0])):
        ax2.text(intrinsicValuation.iloc[0, i], intrinsicValuation.columns[i], "$" + str(round(intrinsicValuation.iloc[0, i], 2)))
    
    ax2.title.set_text("Valoración Intrínseca")
    ax2.legend(loc = "lower right")
    ax2.set_xlabel(r"\$ (USD)")
    
    # Valoración Relativa
    sectorTickers = list(infoNasdaq[infoNasdaq["Sector"] == infoNasdaq[infoNasdaq["Name"] == Company]["Sector"].iloc[0]]["Symbol"])
    #sectorTickers = np.random.choice(sectorTickers, size = 10)
    evSales, evEBITDA, evFCF = [], [], []

    for i in sectorTickers:
        try:
            metrics = cv.key_metrics(i)
            metrics = metrics[["evToSales", "enterpriseValueOverEBITDA", "evToFreeCashFlow"]].iloc[0]
        
            evSales.append(metrics[0])
            evEBITDA.append(metrics[1])
            evFCF.append(metrics[2])
        
        except:
            pass
    
    evSales = np.array(evSales)
    evSales = evSales[(evSales > 0) & (evSales < 100)]
    evEBITDA = np.array(evEBITDA)
    evEBITDA = evEBITDA[(evEBITDA > 0) & (evEBITDA < 100)]
    evFCF = np.array(evFCF)
    evFCF = evFCF[(evFCF > 0) & (evFCF < 100)]

    # Visual 2 (Análisis Fundamental)
    metrics = cv.key_metrics(ticker)
    metrics = metrics[["evToSales", "enterpriseValueOverEBITDA", "evToFreeCashFlow"]].iloc[0]
    
    ax3 = fig.add_subplot(spec[4, 0])
    ax3.hist(evSales, density = True, color = color[0], alpha = 0.5);
    ax3.axvline(np.median(evSales), label = "EV/Sales Sector: " + str(np.round(np.median(evSales), 2)), 
                linestyle = "--", alpha = 1, color = "k")
    ax3.axvline(metrics[0], label = "EV/Sales: " + ticker + ": " +  str(round(metrics[0], 2)), linestyle = "dotted", alpha = 1, color = color[1])
    ax3.title.set_text("Valoración Relativa EV/Sales")
    ax3.legend(loc = 'upper center', bbox_to_anchor = (0.05, -0.20))
    ax3.set_xlabel("EV/Sales")
    
    ax4 = fig.add_subplot(spec[4, 1])
    ax4.hist(evEBITDA, density = True, color = color[0], alpha = 0.5);
    ax4.axvline(np.median(evEBITDA), label = "EV/EBITDA Sector: " + str(np.round(np.median(evEBITDA), 2)), 
                linestyle = "--", alpha = 1, color = "k")
    ax4.axvline(metrics[1], label = "EV/EBITDA: " + ticker + ": " + str(round(metrics[1], 2)), linestyle = "dotted", alpha = 1, color = color[1])
    ax4.title.set_text("Valoración Relativa EV/EBITDA")
    ax4.legend(loc = 'upper center', bbox_to_anchor = (0.05, -0.20))
    ax4.set_xlabel("EV/EBITDA")
    
    ax5 = fig.add_subplot(spec[4, 2])
    ax5.hist(evFCF, density = True, color = color[0], alpha = 0.5);
    ax5.axvline(np.median(evFCF), label = "EV/FCF Sector: " + str(np.round(np.median(evFCF), 2)), 
                linestyle = "--", alpha = 1, color = "k")
    ax5.axvline(metrics[2], label = "EV/FCF " + ticker + ": " + str(round(metrics[2], 2)), linestyle = "dotted", alpha = 1, color = color[1])
    ax5.title.set_text("Valoración Relativa EV/FCF")
    ax5.legend(loc = 'upper center', bbox_to_anchor = (0.05, -0.20))
    ax5.set_xlabel("EV/FCF")

def optionsParams(stockPrices, rf, opType, T, k):
    S_t = stockPrices[-1]
    returns = stockPrices.pct_change().dropna()
    if T == 1:
        r = rf[0] / 100
    elif T == 3:
        r = rf[1] / 100
    else:
        r = rf[2] / 100
        
    T = T / 12
    sigma = np.std(returns) * np.sqrt(252)
    
    d1 = (np.log(S_t / k) + (r + (sigma **2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S_t / k) + (r - (sigma **2) / 2) * T) / (sigma * np.sqrt(T))
    
    if opType == "Call":
        prima = S_t * st.norm().cdf(d1) - k * np.exp(-r * T) * st.norm().cdf(d2)
        delta = st.norm().cdf(d1)
        rho = T * k * np.exp(-r * T) * st.norm().cdf(d2)
        theta = -r * k * np.exp(-r * T) * st.norm().cdf(d2) - 0.5 * sigma / np.sqrt(T) * S_t * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (d1 ** 2))
        
    else:
        prima = k * np.exp(-r * T) * st.norm().cdf(-d2) - S_t * st.norm().cdf(-d1)
        delta = -st.norm().cdf(-d1)
        rho = -T * k * np.exp(-r * T) * st.norm().cdf(-d2)
        theta = r * k * np.exp(-r * T) * st.norm().cdf(-d2) - 0.5 * (sigma / np.sqrt(T)) * S_t * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (d1 ** 2))
        
    gamma = (1 / (S_t * sigma * np.sqrt(T))) * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (d1 ** 2))
    vega = S_t * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (d1 ** 2)) * np.sqrt(T)
        
    return [prima, delta, gamma, rho, vega, theta]
        
def options(Company, end = dt.datetime.today()):
    infoNasdaq = pd.read_csv("files/nasdaq_screener_1650511730482.csv")
    
    # Treasury yields
    startTY = pd.to_datetime(end) - dt.timedelta(30)
    startTY = str(startTY)
    endTY = str(end)
    URL = ("https://financialmodelingprep.com/api/v4/treasury?from=" + startTY[0:10] + "&to=" + endTY[0:10] + "&apikey=a816c18988b47cbfdc9491598c678edd")
    tYields = pd.DataFrame(get_jsonparsed_data(URL))
    riskFree_s = list(tYields.loc[0, ["month1", "month3", "month6"]])

    # Descarga de información (precios históricos)
    ticker = infoNasdaq[infoNasdaq["Name"] == Company]["Symbol"].values[0]
    ticker = ticker.replace(".", "-")
    end = pd.to_datetime(end) + dt.timedelta(1)
    start = end - dt.timedelta(365 * 2)
    stockPrices = yf.download(ticker, start = start, end = end, progress = False)["Adj Close"]
    
    Ks = [stockPrices[-1] * (1 - 0.10), stockPrices[-1] * (1 - 0.05) ,stockPrices[-1], stockPrices[-1] * (1 + 0.05), stockPrices[-1] * (1 + 0.10)]
    Ks = [round(i, 2) for i in Ks]
    params = ["Prima $ (USD)", "Delta", "Gamma", "Rho", "Vega", "Theta"]
    Ts = [1, 3, 6]
    index = pd.MultiIndex.from_product([Ks, Ts, ["Call", "Put"]], names = ["Ejercicio $ (USD)", "Expiración (mes)", "Tipo de Opción"])
    results = [optionsParams(stockPrices, riskFree_s, i, j, k) for k in Ks for j in Ts for i in ["Call", "Put"]]
    optionsDF = pd.DataFrame(results, index = index, columns = params)
    
    return optionsDF, Ks
    
def optionsVisual(Company, OptionsType_Button, OptionsK_Button, OptionsT_Button, end = dt.datetime.today()):
    
    infoNasdaq = pd.read_csv("files/nasdaq_screener_1650511730482.csv")
    
    # Treasury yields
    startTY = pd.to_datetime(end) - dt.timedelta(30)
    startTY = str(startTY)
    endTY = str(end)
    URL = ("https://financialmodelingprep.com/api/v4/treasury?from=" + startTY[0:10] + "&to=" + endTY[0:10] + "&apikey=a816c18988b47cbfdc9491598c678edd")
    tYields = pd.DataFrame(get_jsonparsed_data(URL))
    rf = list(tYields.loc[0, ["month1", "month3", "month6"]])

    # Descarga de información (precios históricos)
    ticker = infoNasdaq[infoNasdaq["Name"] == Company]["Symbol"].values[0]
    ticker = ticker.replace(".", "-")
    end = pd.to_datetime(end) + dt.timedelta(1)
    start = end - dt.timedelta(365 * 2)
    stockPrices = yf.download(ticker, start = start, end = end, progress = False)["Adj Close"]
    
    # Parámetros de la opción
    T = OptionsT_Button.value
    k = OptionsK_Button.value
    S_t = stockPrices[-1]
    returns = stockPrices.pct_change().dropna()
    T = T / 12
    if OptionsT_Button.value == 1:
        r = rf[0] / 100
    elif OptionsT_Button.value == 3:
        r = rf[1] / 100
    else:
        r = rf[2] / 100
        
    sigma = np.std(returns) * np.sqrt(252)
    
    d1 = (np.log(S_t / k) + (r + (sigma **2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S_t / k) + (r - (sigma **2) / 2) * T) / (sigma * np.sqrt(T))
    
    if OptionsType_Button.value == "Call":
        prima = S_t * st.norm().cdf(d1) - k * np.exp(-r * T) * st.norm().cdf(d2)
        
        # Visual: Función de Pago
        x = np.arange(0, 2 * S_t, 0.0001)
        fig1 = go.Figure(data = go.Scatter(x = x, y = [-prima if i < k else i - prima - k for i in x]))
        fig1.update_layout(title = "Función de Pago del Call",
                   xaxis_title = r"$S_T$",
                   yaxis_title = "USD")
        
        # Visual: Evolución del Capital
        stockCloses = yf.download(ticker, start = pd.to_datetime(stockPrices.index[-1]), end = end + dt.timedelta(OptionsT_Button.value * 30), progress = False)["Adj Close"]
        payoff = np.round(stockCloses, 2) - prima - k
        #fig2 = go.Figure(data = go.Scatter(x = stockCloses.index, y = stockCloses.values))
        #fig2.update_layout(title = "Evolución de la Función de Pago del Call", 
        #                  xaxis_title = "Fecha",
        #                  yaxis_title = "USD")
        
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x = stockCloses.index, y = stockCloses.values, name = r"$S_t$"), secondary_y = False,)
        fig2.add_trace(go.Scatter(x = stockCloses.index, y = np.ones(len(stockCloses)) * k, name = r"$K$"), secondary_y = False,)
        fig2.add_trace(go.Scatter(x = stockCloses.index, y = payoff.values, name = r"$C_t$"), secondary_y = True,)
        fig2.update_layout(title = "Evolución de la Función de Pago del Call", xaxis_title = "Fecha", yaxis_title = "USD")
            
    else:
        prima = k * np.exp(-r * T) * st.norm().cdf(-d2) - S_t * st.norm().cdf(-d1)
        
        # Visual: Función de Pago
        x = np.arange(0, 2 * S_t, 0.0001)
        fig1 = go.Figure(data = go.Scatter(x = x, y = [k - i - prima if i < k else -prima for i in x]))
        fig1.update_layout(title = "Función de Pago del Put",
                   xaxis_title = r"$S_T$",
                   yaxis_title = "USD")
        
        # Visual: Evolución del Capital
        stockCloses = yf.download(ticker, start = pd.to_datetime(stockPrices.index[-1]), end = end + dt.timedelta(OptionsT_Button.value * 30), progress = False)["Adj Close"]
        payoff = k - np.round(stockCloses, 2) - prima
        #fig2 = go.Figure(data = go.Scatter(x = stockCloses.index, y = stockCloses.values))
        #fig2.update_layout(title = "Evolución de la Función de Pago del Put", 
        #                  xaxis_title = "Fecha",
        #                  yaxis_title = "USD")
        
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x = stockCloses.index, y = stockCloses.values, name = r"$S_t$"), secondary_y = False,)
        fig2.add_trace(go.Scatter(x = stockCloses.index, y = np.ones(len(stockCloses)) * k, name = r"$K$"), secondary_y = False,)
        fig2.add_trace(go.Scatter(x = stockCloses.index, y = payoff.values, name = r"$P_t$"), secondary_y = True,)
        fig2.update_layout(title = "Evolución de la Función de Pago del Put", xaxis_title = "Fecha", yaxis_title = "USD")
    
    return fig1, fig2, prima
        
      
        