from concurrent.futures.thread import BrokenThreadPool
import requests
from datetime import datetime, timedelta
import json
import pandas as pd
from dateutil.parser import *
import os
import urllib.request
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains
from bs4 import BeautifulSoup
from sqlalchemy.engine.base import ExceptionContextImpl
from sql import Sql
import tweepy as tw
import time
# from requests_ip_rotator import ApiGateway
# import requests_random_user_agent

import concurrent.futures
import os

class Sec():
    def __init__(self) -> None:
        # self.gateway = ApiGateway("https://sec.gov", access_key_id="AKIAZE3X2QM66VQZKUHP", access_key_secret="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")
        # self.gateway.start()
        self.session = requests.Session()
        self.nasdaq_df = pd.read_csv('all_tickers/nasdaq.csv')
        self.nasdaq_df['Name'] = self.nasdaq_df['Name'].str.lower()
        # self.session.mount("https://sec.gov", self.gateway)
        self.db = Sql('sec/sec.db')
        self.user_agent = 'mytupi.com khualinghu@gmail.com'
        self.headers = {
            'user-agent':self.user_agent,

        }

    def get_company(self, cik):
        try:
            url = f'https://sec.report/CIK/{cik}'
            company_html = self.session.get(url=url, headers=self.headers).text
            soup = BeautifulSoup(company_html, features='lxml')
            name = soup.find('div', {'class':'organization-name'}).text
            return name
        except Exception as e: print(e)

    def get_company_details(self,company):
        copy_df = self.nasdaq_df[['Name','Symbol','Market Cap','Country','IPO Year','Volume','Sector','Industry']]
        copy_df['match'] = copy_df.loc[:,'Name'].str.contains(company.lower(), regex=False)
        match_df = copy_df[copy_df['match']==True]
        if match_df.shape[0] > 0:
            symbol = match_df['Symbol'].values[0]
            market_cap = match_df['Market Cap'].values[0]
            country = match_df['Country'].values[0]
            ipo_year = match_df['IPO Year'].values[0]
            volume = match_df['Volume'].values[0]
            sector = match_df['Sector'].values[0]
            industry = match_df['Industry'].values[0]
            return symbol, market_cap, country, ipo_year, volume, sector, industry
        return None,None,None,None,None,None,None

    def get_cik(self, ticker):
        url = 'https://www.sec.gov/files/company_tickers.json'
        tickers_json = self.session.get(url=url, headers=self.headers).json()
        for key in tickers_json.keys():
            if tickers_json[key]['ticker'] == ticker:
                return tickers_json[key]['cik_str']
    
    def search_to_agent_urls_13f(self, startdate, enddate):
        startdate = startdate.isoformat()[:10]
        enddate = enddate.isoformat()[:10]
        page = 0
        all_endpoints = []
        while True:
            page += 1
            start_url = f'https://sec.report/Document/Search//?formType=13F-HR&fromDate={startdate}&toDate={enddate}&page={page}#results'
            start_html = self.session.get(url=start_url, headers=self.headers).text
            soup = BeautifulSoup(start_html, features='lxml')
            a_tags = soup.find_all('a', href=True)
            endpoints = []
            for a_tag in a_tags:
                #newer files in xml
                if ('.xml' in a_tag['href']) or ('.txt' in a_tag['href']):
                    endpoint = a_tag['href']
                    next = a_tag.find_next('small')
                    period_end = next.text[-10:]
                    nextnext = next.find_next('small')
                    cik = nextnext.text.split('(')[-1][4:-1]
                    company_name = nextnext.text.split('(')[0][:-1]
                    nextnextnext = nextnext.find_next('td')
                    date_reported = nextnextnext.text
                    endpoints += [(endpoint,cik,company_name,period_end,date_reported)]

            if len(endpoints) == 0:
                print(page,' pages for ', start_url)
                break
            else:
                all_endpoints += endpoints
                
        return all_endpoints

    def agent_to_xml_urls_13f(self, params):
        url,cik,company_name,period_end,date_reported = params
        endpoints = []
        html = self.session.get(url=url, headers=self.headers).text
        soup = BeautifulSoup(html, features='lxml')

        a_tags = soup.find_all('a', href=True)

        for a_tag in a_tags:
            if ('.xml' in a_tag['href'].lower()) and ('primary' not in a_tag['href'].lower()):
                #format 1
                if a_tag['href'][0] != '/':                    
                    text = a_tag['href']
                    # cik = text.split('/')[4][:10]
                    # company = self.get_company(cik)
                    # # print(company)
                    endpoint = text
                    # endpoints += [(endpoint,cik,company,1)]
                    # if company is None:
                    #     print('no company found', endpoint)
                    endpoints += [(endpoint,cik,company_name,period_end,date_reported,1)]
                #format 2
                else:
                    text = a_tag['href']
                    cik = text.split('/')[3][:10]
                    # company = self.get_company(cik)
                    # print(company)
                    reformat = cik + '-' + text.split('/')[3][10:12] + '-' + text.split('/')[3][12:]
                    endpoint = 'https://sec.report/Document/' + reformat + '/' + text.split('/')[-1]
                    # endpoints += [(endpoint,cik,company,2)]
                    # if company is None:
                    #     print('no company found', endpoint)
                    endpoints += [(endpoint,cik,company_name,period_end,date_reported,2)]
                break
            elif ('.txt' in a_tag['href'].lower()) and ('primary' not in a_tag['href'].lower()):
                #format 1
                if a_tag['href'][0] != '/':                    
                    text = a_tag['href']
                    endpoint = text
                    print(endpoint)
                    endpoints += [(endpoint,cik,company_name,period_end,date_reported,1)]
                #format 2
                else:
                    text = a_tag['href']
                    cik = text.split('/')[4][:10]
                    endpoint = 'https://sec.report' + text
                    endpoints += [(endpoint,cik,company_name,period_end,date_reported,2)]
        if len(endpoints) == 0:
            print(url, 'no endpoints found')
        return endpoints

    def multithreading_agent_to_xml(self,urls):
        endpoints_all = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            for futures in executor.map(self.agent_to_xml_urls_13f, urls):
                endpoints_all += futures
        return endpoints_all

    def xml_to_df_13f1(self, url, ns_num):
        start_html = self.session.get(url=url, headers=self.headers).text
        soup = BeautifulSoup(start_html, features='lxml')
        issuers = [x.text for x in soup.find_all(f'ns{ns_num}:nameofissuer')]
        if url[-3:] == 'txt':
            print(issuers, 'txt')
        #for each issuer, return the company details
        symbols = []
        market_caps = []
        countries = []
        ipo_years = []
        volumes = []
        sectors = []
        industries = []
        for issuer in issuers:
            symbol, market_cap, country, ipo_year, volume, sector, industry = self.get_company_details(issuer)
            symbols += [symbol]
            market_caps += [market_cap]
            countries += [country]
            ipo_years += [ipo_year]
            volumes += [volume]
            sectors += [sector]
            industries += [industry]

        titleclass = [x.text for x in soup.find_all(f'ns{ns_num}:titleofclass')]
        cusip = [x.text for x in soup.find_all(f'ns{ns_num}:cusip')]
        value = [x.text for x in soup.find_all(f'ns{ns_num}:value')]
        sshprnamt = [x.text for x in soup.find_all(f'ns{ns_num}:sshprnamt')]
        sshprnamttype = [x.text for x in soup.find_all(f'ns{ns_num}:sshprnamttype')]
        investmentdiscretion = [x.text for x in soup.find_all(f'ns{ns_num}:investmentdiscretion')]
        votingauthoritysole = [x.text for x in soup.find_all(f'ns{ns_num}:sole')]
        votingauthorityshared = [x.text for x in soup.find_all(f'ns{ns_num}:shared')]
        votingauthoritynone = [x.text for x in soup.find_all(f'ns{ns_num}:none')]
        df = pd.DataFrame({
            'issuers':issuers,
            'ticker':symbols,
            'market_cap':market_caps,
            'country':countries,
            'ipo_year':ipo_years,
            'volume':volumes,
            'sector':sectors,
            'industry':industries,
            'titleclass':titleclass,
            'cusip':cusip,
            'value':value,
            'sshprnamt':sshprnamt,
            'sshprnamttype':sshprnamttype,
            'investmentdiscretion':investmentdiscretion,
            'votingauthoritysole':votingauthoritysole,
            'votingauthorityshared':votingauthorityshared,
            'votingauthoritynone':votingauthoritynone})
        return df

    def xml_to_df_13f2(self, url):
        start_html = self.session.get(url=url, headers=self.headers).text
        soup = BeautifulSoup(start_html, features='lxml')
        issuers = [x.text for x in soup.find_all('nameofissuer')]
        if url[-3:] == 'txt':
            print(issuers, 'txt')
        #for each issuer, return the company details
        symbols = []
        market_caps = []
        countries = []
        ipo_years = []
        volumes = []
        sectors = []
        industries = []
        for issuer in issuers:
            symbol, market_cap, country, ipo_year, volume, sector, industry = self.get_company_details(issuer)
            symbols += [symbol]
            market_caps += [market_cap]
            countries += [country]
            ipo_years += [ipo_year]
            volumes += [volume]
            sectors += [sector]
            industries += [industry]

        titleclass = [x.text for x in soup.find_all('titleofclass')]
        cusip = [x.text for x in soup.find_all('cusip')]
        value = [x.text for x in soup.find_all('value')]
        sshprnamt = [x.text for x in soup.find_all('sshprnamt')]
        sshprnamttype = [x.text for x in soup.find_all('sshprnamttype')]
        investmentdiscretion = [x.text for x in soup.find_all('investmentdiscretion')]
        votingauthoritysole = [x.text for x in soup.find_all('sole')]
        votingauthorityshared = [x.text for x in soup.find_all('shared')]
        votingauthoritynone = [x.text for x in soup.find_all('none')]
        df = pd.DataFrame({
            'issuers':issuers,
            'ticker':symbols,
            'market_cap':market_caps,
            'country':countries,
            'ipo_year':ipo_years,
            'volume':volumes,
            'sector':sectors,
            'industry':industries,
            'titleclass':titleclass,
            'cusip':cusip,
            'value':value,
            'sshprnamt':sshprnamt,
            'sshprnamttype':sshprnamttype,
            'investmentdiscretion':investmentdiscretion,
            'votingauthoritysole':votingauthoritysole,
            'votingauthorityshared':votingauthorityshared,
            'votingauthoritynone':votingauthoritynone})
        return df

    def xml_to_df_to_db(self,endpoints):
        incomplete_endpoints = []
        for each in endpoints:
            endpoint,cik,company_name,period_end,date_reported,format = each
            if format == 1:
                ns_num = 1
                while True:
                    df = self.xml_to_df_13f1(endpoint,ns_num)
                    print(df)
                    if df.shape[0] == 0:
                        if ns_num == 4:
                            break
                        ns_num += 1
                    else:
                        break 
                
                if df.shape[0] == 0: 
                    df = self.xml_to_df_13f2(endpoint)
                    if df.shape[0] == 0:
                        print('error1', endpoint)
                        incomplete_endpoints += [(endpoint,cik,company_name,period_end,date_reported,format)]

            elif format == 2:
                df = self.xml_to_df_13f2(endpoint)
                print(df)
                if df.shape[0] == 0:
                    ns_num = 1
                    while True:
                        df = self.xml_to_df_13f1(endpoint,ns_num)
                        if df.shape[0] == 0:
                            if ns_num == 4:
                                break
                            ns_num += 1
                        else:
                            break
                    if df.shape[0] == 0:
                        print('error2', endpoint)
                        incomplete_endpoints += [(endpoint,cik,company_name,period_end,date_reported,format)]

            if df.shape[0] > 0:
                df['company_cik'] = cik
                df['company_name'] = company_name
                df['period_end'] = period_end
                df['date_reported'] = date_reported
                df['format'] = format
                #save to sql database
                try:
                    self.db.df_to_sql(df,'thirteenf')
                except Exception as e:
                    print(e, endpoint)
                    incomplete_endpoints += [(endpoint,cik,company_name,period_end,date_reported,format)]
            # print(self.db.sql_to_df('thirteenf'))

        with open('sec/incomplete_endpoints.json', 'w') as f:
            json.dump(incomplete_endpoints,f,ensure_ascii=False,indent=4)

    def fetch_13f(self, startdate, enddate):
        urls = self.search_to_agent_urls_13f(startdate,enddate)
        endpoints = self.multithreading_agent_to_xml(urls)
        print('found all endpoints')
        self.xml_to_df_to_db(endpoints)
        n_trys = 0
        while True:
            n_trys += 1
            with open('sec/incomplete_endpoints.json') as f:
                incomplete_endpoints = json.load(f)
            if (len(incomplete_endpoints) == 0):
                break
            elif (n_trys>4):
                print('was not able to complete', incomplete_endpoints)
                break
            self.xml_to_df_to_db(self,incomplete_endpoints)
        # self.gateway.shutdown() 
        
class Finra():
    def __init__(self) -> None:
        self.session = requests.Session()
        pass

    def download_finra_data(self, date, redownload = False):
        date = date.split('-')
        date = date[0] + date[1] + date[2]
        if redownload == False:
            if os.path.isfile(f'short_interest/{date}.txt') == False:
                url = f'https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt'
                file = self.session.get(url)
                with open(f'short_interest/{date}.txt', 'w') as f:
                    f.write(file.text)
        else:
            url = f'https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt'
            file = self.session.get(url)
            with open(f'short_interest/{date}.txt', 'w') as f:
                f.write(file.text)

    def get_df(self, date):
        date = date.split('-')
        date = date[0] + date[1] + date[2]
        df = pd.read_csv(f'short_interest/{date}.txt', sep='|',header=0)
        return df

    def get_si(self, ticker, start_date, end_date, redownload=False):
        delta_t = parse(end_date) - parse(start_date)
        df = pd.DataFrame()
        for day in range(delta_t.days+1):
            try:
                date = str(parse(start_date) + timedelta(days = day)).split()[0]
                self.download_finra_data(date, redownload = redownload)
                df_add = self.get_df(date)
                df_add = df_add[df_add['Symbol']==ticker]
                df = df.append(df_add)
            except: pass
        if df.shape[0]>0:
            df['Date'] = df['Date'].astype(str).apply(parse)
            df['ShortInterest'] = df['ShortVolume']/df['TotalVolume']
        return df
        
class Zacks():
    def __init__(self) -> None:
        pass

    def get_zacks_data(self,ticker):
        output = {}
        try:
            url = f'https://quote-feed.zacks.com/index?t={ticker}'
                
            downloaded_data  = urllib.request.urlopen(url)
            data = downloaded_data.read()
            data_str = data.decode()
            data_json = json.loads(data_str)
            t = data_json[ticker]['updated']
            rank = data_json[ticker]['zacks_rank_text']
            volatility = data_json[ticker]['source']['sungard']['volatility']
            pos_size = data_json[ticker]['source']['sungard']['pos_size']
            yields = data_json[ticker]['source']['sungard']['yield']
            dividend = data_json[ticker]['source']['sungard']['dividend']
            pe_ratio = data_json[ticker]['source']['sungard']['pe_ratio']
            market_cap = data_json[ticker]['source']['sungard']['market_cap']
            earnings = data_json[ticker]['source']['sungard']['earnings'] #- 5.04 EPS in 2010
            volume = data_json[ticker]['source']['sungard']['volume']
            open = data_json[ticker]['source']['sungard']['open']
            close = data_json[ticker]['source']['sungard']['close']
            bid = data_json[ticker]['source']['sungard']['bid']
            ask = data_json[ticker]['source']['sungard']['ask']
            day_high = data_json[ticker]['source']['sungard']['day_high']
            day_low = data_json[ticker]['source']['sungard']['day_low']
            yr_high = data_json[ticker]['source']['sungard']['yrhigh']
            yr_low = data_json[ticker]['source']['sungard']['yrlow']
            output = {'t':t, 'rank':rank, 'volatility':volatility, 'pos_size':pos_size, 'yields':yields,'dividend':dividend, 'pe_ratio':pe_ratio,'market_cap':market_cap,
                'earnings':earnings, 'volume':volume, 'open':open, 'close':close, 'bid':bid, 'ask':ask, 'day_high':day_high, 'day_low':day_low, 'yr_high':yr_high, 'yr_low':yr_low}
        except: pass
        return output

class Nasdaq():
    def __init__(self) -> None:
        self.session = requests.Session()
        self.chrome_options = Options()
        self.chrome_options.binary_location = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary'
        self.ticker_fpath = '/Users/kevinhu/Reddit_Scraping/all_tickers'
        self.chrome_options.add_experimental_option('prefs', {'download.default_directory' : self.ticker_fpath})
        # self.chrome_options.add_argument('window-size=800x841')
        # self.chrome_options.add_argument('headless')
        # self.chrome_options.add_argument('--ignore-certificate-errors')
        # self.chrome_options.add_argument('--allow-running-insecure-content')
        # self.user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.50 Safari/537.36'
        # self.chrome_options.add_argument(f'user-agent={self.user_agent}')
        self.driver = webdriver.Chrome(options=self.chrome_options)
        pass

    def get_earnings(self,ticker):
        try:
            url = f'https://www.nasdaq.com/market-activity/stocks/{ticker}/earnings'
            self.driver.get(url)
            try:
                class1 = WebDriverWait(self.driver, 30).until(EC.presence_of_element_located((By.CLASS_NAME, "module-header")))
            finally: pass
                
            # class1 = self.driver.find_element(By.CLASS_NAME, "module-header")
            table1 = self.driver.find_elements(By.CLASS_NAME, "earnings-surprise__table-cell")
            xpath1 = '/html/body/div[2]/div/main/div[2]/div[4]/div[3]/div/div[1]/div/div[5]/div[1]'
            xpath2 = '/html/body/div[2]/div/main/div[2]/div[4]/div[3]/div/div[1]/div/div[5]/div[2]'
            table2 = self.driver.find_element(By.XPATH, xpath1).find_elements(By.CLASS_NAME,"earnings-forecast__cell")
            table3 = self.driver.find_element(By.XPATH, xpath2).find_elements(By.CLASS_NAME,"earnings-forecast__cell")
            
            #Earnings Date
            try:
                date = parse(class1.text.split(':')[1]).strftime('%Y-%m-%d')
            except:
                date = None

            #EPS Table
            fiscal_q_end = []
            date_reported = []
            eps = []
            eps_forecast = []
            perc_suprise = []

            n_rows = int(len(table1)/5)
            for i in range(n_rows):
                fiscal_q_end += [table1[5*i].text]
                date_reported += [table1[5*i+1].text]
                eps  += [table1[5*i+2].text]
                eps_forecast += [table1[5*i+3].text]
                perc_suprise += [table1[5*i+4].text]
            
            eps_dict = {'fiscal_q_end':fiscal_q_end, 'date_reported':date_reported, 'eps':eps, 'eps_forecast':eps_forecast, 'perc_suprise':perc_suprise}

            #Earnings Forecast Table - Yearly
            fiscal_y_end = []
            eps_forecast = []
            high_eps_forecast = []
            low_eps_forecast = []
            n_estimate = []
            revisions_up = []
            revisions_down = []

            n_rows = int(len(table2)/7)
            for i in range(n_rows):
                fiscal_y_end += [table2[7*i].text]
                eps_forecast += [table2[7*i+1].text]
                high_eps_forecast  += [table2[7*i+2].text]
                low_eps_forecast += [table2[7*i+3].text]
                n_estimate += [table2[7*i+4].text]
                revisions_up += [table2[7*i+5].text]
                revisions_down += [table2[7*i+6].text]

            yforecast_dict = {'fiscal_y_end':fiscal_y_end, 'eps_forecast':eps_forecast, 'high_eps_forecast':high_eps_forecast,
                'low_eps_forecast':low_eps_forecast, 'n_estimate':n_estimate, 'revisions_up':revisions_up, 'revisions_down':revisions_down}

            #Earnings Forecast Table - Quarterly
            fiscal_q_end = []
            eps_forecast = []
            high_eps_forecast = []
            low_eps_forecast = []
            n_estimate = []
            revisions_up = []
            revisions_down = []

            n_rows = int(len(table3)/7)
            for i in range(n_rows):
                fiscal_q_end += [table3[7*i].text]
                eps_forecast += [table3[7*i+1].text]
                high_eps_forecast  += [table3[7*i+2].text]
                low_eps_forecast += [table3[7*i+3].text]
                n_estimate += [table3[7*i+4].text]
                revisions_up += [table3[7*i+5].text]
                revisions_down += [table3[7*i+6].text]

            qforecast_dict = {'fiscal_q_end':fiscal_q_end, 'eps_forecast':eps_forecast, 'high_eps_forecast':high_eps_forecast,
                'low_eps_forecast':low_eps_forecast, 'n_estimate':n_estimate, 'revisions_up':revisions_up, 'revisions_down':revisions_down}

            todays_date = datetime.today().strftime('%Y-%m-%d')
            output = {'earnings_date':date, 'eps_dict':eps_dict, 'yforecast_dict':yforecast_dict, 'qforecast_dict':qforecast_dict}
            
            if os.path.isfile(f'nasdaq/earnings/{ticker}.json') == True:
                with open(f'nasdaq/earnings/{ticker}.json') as f:
                    final = json.load(f)
                final[todays_date] = output
            else:
                final = {todays_date:output}
            
            with open(f'nasdaq/earnings/{ticker}.json', 'w') as f:
                json.dump(final, f, ensure_ascii=False, indent=4)

            return ticker, output
        except Exception as e:
            print(e)
            return ticker, None

    def get_earnings_all(self, tickers):
        output = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for futures in executor.map(self.get_earnings, tickers):
                ticker, dict = futures
                output[ticker] = dict
        self.driver.quit()
        return output

    def download_tickers(self):
        url = f'https://www.nasdaq.com/market-activity/stocks/screener'
        self.driver.get(url)
        xpath = '/html/body/div[2]/div/main/div[2]/article/div[3]/div[1]/div/div/div[3]/div[2]/div[2]/div/button'
        try:
            button = WebDriverWait(self.driver, 30).until(EC.presence_of_element_located((By.XPATH,xpath)))
        finally: pass
        action = ActionChains(self.driver).click(button)
        action.pause(3)
        action.perform()
        self.driver.quit()
        files = ['all_tickers/'+f for f in os.listdir('all_tickers') if not f.startswith('.')]
        latest = max(files , key = os.path.getctime)
        os.rename(latest, 'all_tickers/nasdaq.csv')

class Twitter():
    def __init__(self) -> None:
        api_key = 'eTlwEw1MOmaZtyLlx0wjIdb1F'
        secret_api_key = '75JmPT9f7G47n7GfozxnFzrQajhWkFPm4XLurIJ04MCbDiQ5tm'
        bearer_token = 'AAAAAAAAAAAAAAAAAAAAAJVTVwEAAAAAVe5ikhXmW4Pps%2FBum8y6zBc%2Bsbo%3Dt1TVHgWDcc3dASBn6OS84uxZjtWTaMGbixWnrLqQg34XA4A7Oz'
        access_token = '1440036104283906062-wlMG3VJzf4KMKBDiZNJWnsdbJoG9WM'
        access_token_secret = 'Qj68X7UOqGmJVLQI0okL2GbtgLB4tgzlW7fnnp02LSuNj'
        
        callback_uri = 'oob'
        auth = tw.OAuthHandler(api_key,secret_api_key,callback_uri)
        # redirect_url = auth.get_authorization_url()
        # print(redirect_url)
        # user_pin_input = input('user pin')
        # auth.get_access_token(user_pin_input)
        # print(auth.access_token, auth.access_token_secret)
        api = tw.API(auth)
        me = api.me()
        print(me.screen_name)

        auth.set_access_token(access_token,access_token_secret)
        api = tw.API(auth)
        me = api.me()
        print(me.screen_name)

        # for status in tw.Cursor(api.user_timeline, screen_name='@realDonaldTrump', tweet_mode="extended").items():
        #     print(status.full_text)

        # try:
        #     elons_tweets = api.user_timeline(screen_name='@elonmusk')
        # except tw.Forbidden as e:
        #     print(e.response.json())
        # for tweet in elons_tweets:
        #     print(tweet.text)

if __name__ == '__main__':
    # inst = Short_Interest()
    # start_date = str(datetime.today()-timedelta(days = 5)).split()[0]
    # end_date = str(datetime.today()-timedelta(days = 4)).split()[0]
    # df = inst.get_si('TSLA',start_date,end_date)
    # print(df)
    # zacks = Zacks()
    # dict = zacks.get_zacks_data('TSLA')
    # print(dict)
    # nasdaq = Nasdaq()
    # tickers = ['TSLA', 'BYDDY', 'PTON', 'LCID', 'BNTX', 'MRNA']
    # output = nasdaq.get_earnings_all(tickers)
    # print(output)
    # nasdaq.download_tickers()

    # sec = Sec()
    # start = datetime.today()-timedelta(days=365)
    # end = datetime.today()
    # now = time.time()
    # sec.fetch_13f(start,end)
    # print(time.time()-now, 'seconds to complete')

    api_key = 'b2kJZa8NR0HZwVFnOGvE7nSJw'
    secret_api_key = 'o4IReLBmbB5H3U7BQLS3g1tW3PHsFx5EzS8V7WSR9pKMkSV9ZJ'
    bearer_token = 'AAAAAAAAAAAAAAAAAAAAAJVTVwEAAAAAT0t3pE3%2B1lI8JLvAnmhKzgOI3iw%3DaMvZfTODwCtOUVbZaF5JPYzsgiBkogAi734pHuDPTSLG1kBWrQ'
    access_token = '1440036104283906062-7e6gr2qRJkUhv3ibPvIxnLW82eleHA'
    access_token_secret = 'qSjGObtrrG8pM9erXqNe0CrG2c5KdgMYCUAsXjRa8kb2g'

    auth = tw.OAuthHandler(api_key,secret_api_key)

    auth.set_access_token(access_token,access_token_secret)
    auth.secure = True
    api = tw.API(auth, wait_on_rate_limit=True)
    search_words = "#sexualharassment"
    date_since = "2020-07-14"
    tweets = tw.Cursor(api.search_tweets,
              q=search_words,
              lang="en").items(1000)
    for tweet in tweets:
        print(tweet.text)