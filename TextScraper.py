import bs4 as bs
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from selenium.webdriver.chrome.service import Service 
import requests
import random
from bs4 import BeautifulSoup
import pandas as pd
import os
from datetime import datetime

link='http://www.kremlin.ru/'  

chrome_path = "/Users/test/anaconda3/envs/Masterthesis/lib/python3.8/site-packages/chromedriver_py/chromedriver_mac-x64"#/usr/local/bin/chromedriver" #ajust this to precise chromedriver if runing this code on other computer 
 

filename = 'working_memory.csv'
if os.path.isfile(filename): #ingestion of list_of_links, the product of scraping with the "linkscraper" 
    list_of_links = pd.read_csv('working_memory.csv', encoding='utf-8-sig')  

    
else:
    list_of_links = pd.read_csv('list_ot_links-ukraine-original copy.csv')
    
list_of_links = list(list_of_links['link'])


print(list_of_links)


screen_sizes = [[1366,768], [1920,1080]]

countries = ['KZ', 'RU', 'UZ', 'BY'] # remove

def get_proxy(): #scraping of a list of proxies from 'https://sslproxy.org' to later aid in getting around the kremlin.ru web scraping protections
    response = requests.get('https://sslproxies.org/')
    soup = BeautifulSoup(response.text, 'lxml')
    tag = 'textarea'
    proxies = soup.find_all(tag)
    proxies = [i for i in proxies if ':' in i][1:]
    return proxies


def set_driver(proxies=False):
    chrome_options = webdriver.ChromeOptions()
    if proxies != False:
        if len(proxies) > 0:
            chrome_options.add_argument('--proxy-server={}'.format(proxies[0]))
            proxies.pop(0)
        else:
            proxies = get_proxy()
            chrome_options.add_argument('--proxy-server={}'.format(proxies[0]))
    else:
        proxies = get_proxy()
        chrome_options.add_argument('--proxy-server={}'.format(proxies[0]))
        proxies.pop(0)
    #print('proxies added', proxies)
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument('--profile-directory=Default')
    chrome_options.add_argument("--incognito")
    chrome_options.add_argument("--disable-plugins-discovery")
    chrome_options.add_argument("--start-maximized")
    # chrome_options.headless = True
    chrome_options.add_argument("--disable-gpu")


    chrome_options.add_argument("enable-automation")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-browser-side-navigation")
    # chrome_options.setPageLoadStrategy(PageLoadStrategy.NORMAL)
    chrome_options.add_argument("--use-mock-keychain")
    chrome_options.add_argument("--disable-features=MediaRouter,ProcessPerSiteUpToMainFrameThreshold")


    driver = webdriver.Chrome(chrome_path, options=chrome_options)
    driver.delete_all_cookies()
    # driver.set_window_size(1000, 800)
    # driver.set_window_position(0, 0)
    screen = random.choice(screen_sizes)
    driver.set_window_size(screen[0], screen[1])
    chrome_options.add_argument("–disable-dev-shm-usage")
    chrome_options.add_argument("start-maximized")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-setuid-sandbox")
    # chrome_options.add_argument("referer=" + category_page)
    chrome_options.add_argument('accept-encoding=' + 'gzip, deflate, br')
    chrome_options.add_argument(
        'accept=' + 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9')
    chrome_options.add_argument("upgrade-insecure-requests=" + "1")
    chrome_options.add_argument('cookies=' + 'yes')
    chrome_options.add_argument("timezone=" + "-360")
    chrome_options.add_argument("languages-js=" + "pt-BR,en-US,pt,en")
    chrome_options.add_argument(
        "plugins=" + "Plugin 0: Chrome PDF Plugin; Portable Document Format; internal-pdf-viewer. Plugin 1: Chrome PDF Viewer; ; mhjfbmdgcfjbbpaeojofohoefgiehjai. Plugin 2: Native Client; ; internal-nacl-plugin. ")
    chrome_options.add_argument("screen_depth=" + "24")
    chrome_options.add_argument("screen_left" + "0")
    chrome_options.add_argument("screen_top" + "0")
    chrome_options.add_argument('accept-language' + 'pt-BR,pt;q=0.9,en-US,en;q=0.8')
    connected = False
    while not connected:
 
        try: 
            driver = webdriver.Chrome(chrome_path, options=chrome_options)
            connected = True
        except (NoSuchElementException, TimeoutException, WebDriverException) as e:
            pass
    return driver, proxies


def try_proxy(driver, proxies, link, list_of_links):
    soup = ''
    page = ''
    get_articles(driver, link, proxies, list_of_links)
    return driver, soup, proxies, page



def get_articles(driver, link, proxies, list_of_links):
    driver.get(link)

    articles_link = 'http://www.kremlin.ru/events/president/news/'
    now = str(datetime.now())
    csvname = str("/Users/test/Desktop/Articles/" + now + "articles.csv")

    driver, proxies = set_driver(proxies)
    df = pd.DataFrame()
    while 0 < len(list_of_links):

        i = list_of_links[0]
        print(i)
        try:
            try:
                driver.get(articles_link + str(i))
            except:
                driver.get(articles_link + str(i))
            driver.execute_script("window.stop();")
            page = driver.page_source
            soup = bs.BeautifulSoup(page, 'html.parser')
            try:
                text = soup.find("article").text
                if 'Forbidden' in text:
                    try:
                        driver, proxies = set_driver(proxies)
                        #print(i)
                        driver.get(articles_link + str(i))
                        driver.execute_script("window.stop();")

                        page = driver.page_source
                        soup = bs.BeautifulSoup(page, 'html.parser')

                        try:
                            text = soup.find("article").text
                        except:
                            text = ''
                    except:
                        text = ''

            except:
                try:
                    driver, proxies = set_driver(proxies)
                    try:
                        driver.get(articles_link + str(i))
                    except:
                        driver.get(articles_link + str(i))
                    driver.execute_script("window.stop();")
                    page = driver.page_source
                    soup = bs.BeautifulSoup(page, 'html.parser')

                    try:
                        text = soup.find("article").text
                    except:
                        text = ''
                except:
                    text = ''

            try:
                date_of_article = soup.find("time").attrs['datetime']
            except:
                date_of_article = ''

            infodict = {'id': i, 'date': date_of_article, 'text': text }


        except:
            driver, proxies = set_driver(proxies)
            driver.get(articles_link + str(i))
            driver.execute_script("window.stop();")

            page = driver.page_source
            soup = bs.BeautifulSoup(page, 'html.parser')

            try:
                text = soup.find("article").text
            except:
                text = ''

            try:
                date_of_article = soup.find("time").attrs['datetime']
            except:
                date_of_article = ''



            infodict = {'id': i, 'date': date_of_article, 'text': text }


        if infodict['date']=='':
            try:
                driver.get(articles_link + str(i))
                driver.execute_script("window.stop();")

                page = driver.page_source
                soup = bs.BeautifulSoup(page, 'html.parser')
                try:
                    text = soup.find("article").text
                except:
                    text = ''

                try:
                    date_of_article = soup.find("time").attrs['datetime']
                except:
                    date_of_article = ''


                infodict = {'id': i, 'date': date_of_article, 'text': text }

            except:
                driver, proxies = set_driver(proxies)
                tt = 0
                while tt==0:
                    try:
                        driver.get(articles_link + str(i))
                        tt=1
                    except:
                        pass
                driver.execute_script("window.stop();")
                page = driver.page_source
                soup = bs.BeautifulSoup(page, 'html.parser')

                try:
                    text = soup.find("article").text
                except:
                    text = ''

                try:
                    date_of_article = soup.find("time").attrs['datetime']
                except:
                    date_of_article = ''
                    
                infodict = {'id': i, 'date': date_of_article, 'text': text }

        print(infodict)            
        if infodict["text"] == '':
            list_of_links.append(infodict["id"])
        else:
            df_dict = pd.DataFrame([infodict])
            df = pd.concat([df, df_dict], ignore_index=True)
            df.to_csv(csvname, index=False) 

        list_of_links.pop(0)
        with open(filename, 'w') as file:
            for item in list_of_links:
                file.write(str(item) + '\n')
        with open(str("/Users/test/Desktop/Backupmemory" + str(now) + filename), 'w') as file:
            for item in list_of_links:
                file.write(str(item) + '\n')
        

    return df

proxies = get_proxy()
driver, proxies = set_driver(proxies)

try_proxy(driver, proxies, link, list_of_links)
