#%%
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Sentiment_Indicator.distilroberta import get_sentiment

def get_news_links(tickers):
    options = Options()
    options.headless = True
    options.add_argument('--headless')
    options.binary_location = r'C:\Users\Tristan\Desktop\Projects\chrome-win64\chrome.exe'
    service = Service('C:\\Users\\Tristan\\Desktop\\Projects\\chromedriver-win64\\chromedriver.exe')
    driver = webdriver.Chrome(service=service, options=options)
    ticker_news_data = {}
    for ticker in tickers:
        driver.get(f'https://www.stockwatch.com/Quote/Detail?U:{ticker}')
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        rows = soup.find_all('tr')
        news_data = []
        
        for row in rows:
            td_elements = row.find_all('td')
            if len(td_elements) > 0:
                news_date = td_elements[0].get_text(strip=True)
                news_td = row.find('td', class_='gt-largefont')
                if news_td:
                    a_element = news_td.find('a')
                    if a_element and 'href' in a_element.attrs:
                        news_link = 'https://www.stockwatch.com' + a_element['href']
                        news_date = pd.to_datetime(news_date).date()
                        news_data.append({'date': news_date, 'link': news_link})

        ticker_news_data[ticker] = news_data
    driver.quit()
    return ticker_news_data    
# %%
def get_news_text(news_links):
    options = Options()
    options.headless = True
    options.add_argument('--headless')
    options.binary_location = r'C:\Users\Tristan\Desktop\Projects\chrome-win64\chrome.exe'
    service = Service('C:\\Users\\Tristan\\Desktop\\Projects\\chromedriver-win64\\chromedriver.exe')
    driver = webdriver.Chrome(service=service, options=options)
    for ticker in news_links:
        for news in news_links[ticker]:
            driver.get(news['link'])
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            news_span = soup.find('span', id='MainContent_NewsText')
            if news_span:
                text = news_span.get_text(strip=True)
                news['text'] = text
            else:
                news['text'] = 'No content found'
    driver.quit()
    return news_links
#%%
def get_news_sentiment(news_links):
    for ticker in news_links:
        for news in news_links[ticker]:
            sentiment = get_sentiment(news['text'])
            news['sentiment'] = sentiment
    return news_links
# %%
