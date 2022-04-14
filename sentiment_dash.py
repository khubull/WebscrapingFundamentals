import json
import os

import dash
from dash.html.Data import Data
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dash_table as dt
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc 
from reddit import Reddit_Monitor
import time
from dateutil.parser import *
import pytz
from nlp import ApeSpeech
from alpaca_api import Alpaca_API
from datetime import date, datetime, timedelta, timezone
from fundamental_scraping import Finra, Zacks, Nasdaq
import curve_fitting

class Sentiment_Dash():
    def __init__(self) -> None:
        self.redditor = ''
        self.redditors = [] 
        self.dates = []
        self.leaderboard = pd.DataFrame(columns=['author','score','win_ratio'])
        self.userdata = pd.DataFrame(columns=['ticker', 'text','direction','gain','time','score','int','wins'])
        self.newcomments = pd.DataFrame(columns=['ticker','time','direction','redditor','rank','text'])
        self.candles = pd.DataFrame()
        self.reddit = Reddit_Monitor()
        self.finra = Finra()
        self.zacks = Zacks()
        self.api = Alpaca_API(quote_fpath='dash_stream/quotes.json')
        self.tradable_tickers = pd.read_csv('all_tickers/nasdaq.csv')['Symbol'].to_list()

    def dates_available(self):
        pred_files = ['predictions/'+f for f in os.listdir('predictions') if not f.startswith('.')]
        dates = list(set([file.split('_')[0].split('/')[1] for file in pred_files]))
        return dates

    def leaderboard_table(self, dates):
        #1. Import leaderboard json
        authors = []
        scores = []
        win_ratios = []
        for date in dates:
            try:
                with open(f'predictions/{date}_leaderboard2.json') as f:
                    prev_preds = json.load(f)
                for i in prev_preds.keys():
                    authors += [prev_preds[i]['author']]
                    scores += [prev_preds[i]['tot_score']]
                    win_ratios += [prev_preds[i]['win_ratio']]
            except: pass
        leaderboard = pd.DataFrame({'author':authors, 'score':scores, 'win_ratio':win_ratios})
        leaderboard[['score', 'win_ratio']] = leaderboard[['score', 'win_ratio']].astype(float)
        leaderboard.sort_values(by=['win_ratio', 'score'], inplace=True, ascending=False)
        leaderboard.reset_index(drop=True,inplace=True)
        self.leaderboard = leaderboard.head(10)
        self.redditors = self.leaderboard['author'].tolist()
    
    def load_user_comments(self, user, dates):
        user_data = {'ticker':[],'text':[],'direction':[],'gain':[],'time':[]}
        for date in dates:
            try:
                with open(f'predictions/{date}_leaderspast.json') as f:
                    comments_json = json.load(f)
                if user in comments_json.keys():
                    user_data['ticker'] += comments_json[user]['ticker']
                    user_data['text'] += comments_json[user]['text']
                    user_data['direction'] += comments_json[user]['direction']
                    user_data['gain'] += comments_json[user]['gain']
                    user_data['time'] += comments_json[user]['time']
            except: pass
        return user_data

    def update_userdata(self, leaderboard_row):
        row = leaderboard_row[0]
        user = self.leaderboard.loc[row,'author']
        self.redditor = user
        userdata_dict = self.load_user_comments(user,self.dates)
        df = pd.DataFrame(userdata_dict)
        df['score'] = df['direction'] * df['gain']
        df.loc[(df['score'] > 0, 'wins')] = 'win'
        df.loc[(df['score'] < 0, 'wins')] = 'loss'
        df.loc[(df['score'] == 0, 'wins')] = 'n/a'
        df['int'] = 1
        df.drop_duplicates(['text'],keep='last',inplace=True)
        df.reset_index(drop=True,inplace=True)
        self.userdata = df
        
    def build_leaderboard(self):
        return dt.DataTable(
            id = 'select-user',
            columns=[{"name":i,"id":i} for i in self.leaderboard.columns],
            data= self.leaderboard.to_dict('records'),
            style_cell={'textAlign': 'center', 'font-family':'verdana'},
            style_as_list_view=True,
            row_selectable='single',
            selected_rows=[0],
            persistence = True,
            persistence_type = 'memory',
            style_header={ 'border': '1px solid blue'}
            )

    def build_comments_query(self):
        yesterday = datetime.today()-timedelta(days=1)
        yesterday_close = yesterday.strftime('%Y-%m-%d') + ' 16:00:00'
        return dbc.Input(
            id = 'time-filter',
            placeholder=yesterday_close,
            type='text',
            value=yesterday_close,
            persistence = True,
            persistence_type = 'memory'
            )

    def build_ticker_query(self):
        ids = ['ticker', 'start-time', 'end-time']
        placeholders = ['ticker', '00:00', '00:00']
        current_time = str(datetime.now()-timedelta(minutes=15)).split()[1][:5]
        initial_vals = ['SPY', '09:00', current_time]
        classname = ['one columns', 'one columns', 'one columns']
        return dbc.Col([
                dbc.InputGroup(
                    [dbc.Input(
                        id = id,
                        placeholder=placeholders[i],
                        type='text',
                        value=initial_vals[i],
                        persistence = True,
                        persistence_type = 'memory') for i,id in enumerate(ids)]+[
                    dbc.RadioItems(
                        id = 'granularity',
                        options=[
                            {'label': '1Min', 'value': '1Min'},
                            {'label': '5Min', 'value': '5Min'},
                            {'label': '15Min', 'value': '15Min'},
                            {'label': '30Min', 'value': '30Min'},
                            {'label': '60Min', 'value': '60Min'},
                            {'label': '1Day', 'value': '1Day'},
                            {'label': '2Day', 'value': '2Day'}
                        ],
                        value='15Min',
                        inline=True
                    )]
                ),
                dbc.Col([
                    dcc.DatePickerRange(
                        id = 'date-range',
                        month_format='MMMM Y',
                        start_date=datetime.today().strftime('%Y-%m-%d'),
                        end_date=datetime.today().strftime('%Y-%m-%d'),
                        persistence = True,
                        persistence_type = 'memory'
                    )
                ])
            ])
    
    def build_dates_dropdown(self):
        dates = self.dates_available()
        dates = ['show_all'] + dates
        return dcc.Dropdown(
            id='select-date',
            options=[{'label': i, 'value': i} for i in dates],
            value=dates[0],
            multi=False,
            persistence = True,
            persistence_type = 'session',
            searchable=False,
            style={'height':'50px'}
            )

    def toggle_stream(self):
        return dbc.Switch(
            id='start-stream',
            label='Stream On/Off'
            )

    def load_quotes(self):
        if os.path.isfile('dash_stream/quotes.json'):
            with open('dash_stream/quotes.json','r') as f:
                quotes = json.load(f)
            # if quotes is not None:
            #     for ticker in quotes.keys():
            #         quotes[ticker]['time'] = quotes[ticker]['time'][-10:]
            #         quotes[ticker]['price'] = quotes[ticker]['price'][-10:]

            return quotes

    def build_app(self, app):
        app.layout = dbc.Container(
            [
                dcc.Interval(id='comments-interval', interval=1000*30),
                dcc.Interval(id='quotes-interval', interval=1000*1),
                dbc.Row([
                    html.H1('Reddit Dashboard'),
                    dbc.Col([
                        html.H3('Leaderboard'),
                        dbc.Col(self.build_dates_dropdown()),
                        dbc.Col(id = 'leaderboard'),
                        dbc.Col([
                            html.H3('Recent Comments'),
                            html.Div(self.build_comments_query()),
                            html.Div(id = 'new-comments'),
                            # html.Div(id = 'tickers-mentioned-animation')
                        ]),
                    ], width=5),
                    dbc.Col([
                        dbc.Row(id = 'pie-chart1'),
                        dbc.Row(id = 'pie-chart2')
                    ], width=2),
                    dbc.Col([
                        html.H3('Previous Comments'),
                        html.Div(id = 'comments')
                    ], width=5),
                ]),
                dbc.Row([
                    html.H1('Stock Screener'),
                    self.toggle_stream(),
                ]),
                dbc.Row([
                    dbc.InputGroup([
                        dbc.Button('Load Candles', id='load-candles', n_clicks=0, color="primary", className="me-1"),
                        dbc.Button('Download Earnings', id ='download-earnings', n_clicks=0, color="success", className="me-1"),
                        dbc.Button('Save Tickers', id='save-tickers-button', n_clicks=0, color="success", className="me-1"),
                        dbc.Button('Clear Tickers', id='clear-tickers-button', n_clicks=0, color="danger", className="me-1"),
                    ]),
                    dbc.Col(self.build_ticker_query(), width=6),
                    dbc.Col(id = 'update-earnings'),
                    html.Div(id='save-tickers'),
                    html.Div(id='stream-quotes'),
                ]),
                dbc.Row([
                    dbc.Col(id = 'zacks', width=1),
                    dbc.Col(id = 'earnings', width=7),
                    dbc.Col(id = 'short-interest',width=4)
                ]),
                dbc.Row([
                    dbc.Col(id = 'get-candles', width=5),
                    dbc.Col(id='live-stream-quotes', width=5),
                ]),
                dbc.Row([
                    dbc.Button('Load 13F', id='load-hist-button', n_clicks=0, color="primary", className="me-1"),
                    dbc.Col(id = 'institutional-hist', width=5),
                ])
            ],
            className="p-5",
            fluid = True
        )      

        @app.callback(Output('leaderboard', 'children'), Input('select-date', 'value'))
        def leaderboard(date):
            if date == 'show_all':
                dates = self.dates_available()
                self.dates = dates
            else:
                self.dates = [date]

            self.leaderboard_table(self.dates)
            
            if self.leaderboard.shape[0] > 0:
                return dt.DataTable(
                    id = 'select-user',
                    columns=[{"name":i,"id":i} for i in self.leaderboard.columns],
                    data=self.leaderboard.to_dict('records'),
                    style_cell={'textAlign': 'center', 'font-family':'verdana'},
                    style_as_list_view=True,
                    row_selectable='single',
                    selected_rows=[0],
                    persistence = True,
                    persistence_type = 'memory',
                    style_header={ 'border': '1px solid blue'}
                    )

        @app.callback(Output('tickers-mentioned-animation', 'children'), Input('quotes-interval', 'n_intervals'))
        def ticker_animation(interval):
            file_list = ['jsons/stream/'+f for f in os.listdir('jsons/stream') if not f.startswith('.')]
            latest_stream = max(file_list , key = os.path.getctime)
            df = self.reddit.json_to_df(latest_stream, min_mentions=3)
            if df.shape[0]>0:
                fig = self.reddit.show_animation(df)
                return dcc.Graph(figure=fig)

        @app.callback(Output('pie-chart1', 'children'), Input('select-user', 'selected_rows'))
        def winning_pie(leaderboard_row):
            self.update_userdata(leaderboard_row)
            fig = px.pie(self.userdata,names=self.userdata['wins'], values=self.userdata['int'], hole=0.6)
            fig.update_traces(textposition='inside', textinfo='value+label')
            fig.update_layout(showlegend=False, margin={'b':0,'l':0,'r':0,'t':0}, title = {'text': "WinLoss",'font':{'size':25},'y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'})
            return dcc.Graph(figure=fig)

        @app.callback(Output('pie-chart2', 'children'), Input('select-user', 'selected_rows'))
        def ticker_pie(leaderboard_row):
            self.update_userdata(leaderboard_row)
            fig = px.pie(self.userdata,names=self.userdata['ticker'], values=self.userdata['int'],hole=0.6)
            fig.update_traces(textposition='inside', textinfo='value+label')
            fig.update_layout(showlegend=False, margin={'b':0,'l':0,'r':0,'t':0}, title = {'text': "Tickers",'font':{'size':25},'y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'})
            return dcc.Graph(figure=fig)

        @app.callback(Output('comments', 'children'), Input('select-user', 'selected_rows'))
        def comments(leaderboard_row):
            self.update_userdata(leaderboard_row)
            return dt.DataTable(
                columns=[{"name":i,"id":i} for i in self.userdata[['ticker','time','gain','direction','wins','text']]],
                data=self.userdata.to_dict('records'),
                style_cell={'textAlign': 'center', 'font-family':'verdana'},
                style_as_list_view=True,
                persistence = True,
                persistence_type = 'memory',
                style_header={ 'border': '1px solid blue'},
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    },
                style_data_conditional=[
                    {
                    'if': {
                        'filter_query': '{score} > 0',
                        'column_id': 'wins'
                    },
                    'backgroundColor': 'Green',
                    'color': 'white'
                    },
                    {
                    'if': {
                        'filter_query': '{score} = 0',
                        'column_id': 'wins'
                    },
                    'backgroundColor': '',
                    'color': 'white'
                    },
                    {
                    'if': {
                        'filter_query': '{score} < 0',
                        'column_id': 'wins'
                    },
                    'backgroundColor': 'Red',
                    'color': 'white'
                    }]
                )
        
        @app.callback(Output('new-comments', 'children'), Input('time-filter', 'value'), Input('comments-interval', 'n_intervals'))
        def new_comments(start_time, interval):
            if (start_time is not None):
                for rank, redditor in enumerate(self.redditors):
                    user_comments = self.reddit.get_user_comments(redditor, limit=5)
                    init = time.time()
                    for comment in user_comments:
                        now = time.time()
                        if (now-init) > 1:
                            break
                        text = comment.body
                        comtime = comment.created_utc
                        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(comtime)))
                        if t > start_time:
                            translator = ApeSpeech(text)
                            tickers = translator.find_ticker()
                            for ticker in tickers:
                                direction = translator.get_direction()
                                self.newcomments.loc[self.newcomments.shape[0]] = [ticker, t, direction, redditor, rank+1, text]
                self.newcomments.drop_duplicates(['text'],keep='first',inplace=True)
                self.newcomments =  self.newcomments[self.newcomments['time'] > start_time]
                self.newcomments.sort_values(by=['time'], ascending=False, inplace=True)
                self.newcomments.reset_index(drop=True, inplace=True)
                return dt.DataTable(
                    columns=[{"name":i,"id":i} for i in self.newcomments.columns],
                    data=self.newcomments.to_dict('records'),
                    style_cell={'textAlign': 'center', 'font-family':'verdana'},
                    style_as_list_view=True,
                    persistence = True,
                    persistence_type = 'memory',
                    style_header={ 'border': '1px solid red'},
                    style_data={
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        },
                    style_data_conditional=[
                        {
                        'if': {
                            'filter_query': '{direction} > 0',
                            'column_id': 'direction'
                        },
                        'backgroundColor': 'Green',
                        'color': 'white'
                        },
                        {
                        'if': {
                            'filter_query': '{direction} = 0',
                            'column_id': 'direction'
                        },
                        'backgroundColor': '',
                        'color': 'white'
                        },
                        {
                        'if': {
                            'filter_query': '{direction} < 0',
                            'column_id': 'direction'
                        },
                        'backgroundColor': 'Red',
                        'color': 'white'
                        }]
                    )

        @app.callback(Output('zacks', 'children'), Input('ticker', 'value'))
        def zacks(ticker):
            metrics_dict = self.zacks.get_zacks_data(ticker)
            if 'rank' in metrics_dict.keys():
                t = metrics_dict['t']
                rank = metrics_dict['rank']
                earnings = metrics_dict['earnings']
                market_cap = metrics_dict['market_cap']
                volatility = metrics_dict['volatility']
                volume = metrics_dict['volume']
                yields = metrics_dict['yields']

                df = pd.DataFrame(metrics_dict, index=[0]).transpose()
                df.reset_index(drop=False, inplace=True)
                df.rename(columns={'index':'metric', 0:'value'},inplace=True)
                dict = df.to_dict(orient='records')

                if len(rank) == 0:
                    rank = 'N/A'
                    color = 'gray'
                elif 'buy' in rank.lower():
                    color = 'green'
                elif 'hold' in rank.lower():
                    color = 'gray'
                else:
                    color = 'red'

                return dbc.Row(dbc.Col(f'Zacks Rank: {rank}', style = {'border': f'5px outset {color}', 'backgroundColor': f'{color}', 'fontSize': 20})),

        @app.callback(Output('earnings', 'children'), Input('ticker', 'value'), Input('update-earnings', 'children'))
        def earnings(ticker, update):
            earnings_date = 'N/A'
            eps_suprise = 'N/A'
            eps_forecast = 'N/A'
            analyst_sentiment_nextq = 'N/A'
            color = 'gray'
            color1 = 'gray'
            color2 = 'gray'
            color3 = 'gray'
            try:
                
                if os.path.isfile(f'nasdaq/earnings/{ticker}.json') == True:
                    with open(f'nasdaq/earnings/{ticker}.json') as f:
                        earnings_dict = json.load(f)

                    dates_scraped = list(earnings_dict.keys())
                    latest_date_scrape = max(dates_scraped)
                    try:
                        earnings_date = earnings_dict[latest_date_scrape]['earnings_date']

                        if earnings_date is not None:
                            color = 'green'
                        else:
                            earnings_date = 'N/A'
                            color = 'gray'
                    except: pass

                    #most recent eps
                    try:
                        latest_report_date = earnings_dict[latest_date_scrape]['eps_dict']['date_reported'][0]
                    except:pass
                    try:
                        eps_recent = float(earnings_dict[latest_date_scrape]['eps_dict']['eps'][0])
                    except: pass

                    #most recent eps suprise
                    try:
                        eps_suprise = float(earnings_dict[latest_date_scrape]['eps_dict']['perc_suprise'][0])
                        if eps_suprise < 0:
                            color1 = 'red'
                        elif eps_suprise > 0:
                            color1 = 'green'
                        else:
                            color1 = 'gray'
                    except: pass

                    #over last 4 weeks, number of revisions up for next fiscal quarter
                    try:
                        analyst_sentiment_nextq = int(earnings_dict[latest_date_scrape]['qforecast_dict']['revisions_up'][0])-int(earnings_dict[latest_date_scrape]['qforecast_dict']['revisions_down'][0])
                        if analyst_sentiment_nextq < 0:
                            color2 = 'red'
                        elif analyst_sentiment_nextq > 0:
                            color2 = 'green'
                        else:
                            color2 = 'gray'
                    except: pass

                    #eps forecast for this quarter, better or worse than last quarter
                    try:
                        eps_forecast = float(earnings_dict[latest_date_scrape]['qforecast_dict']['eps_forecast'][0])
                        prev_forecast = float(earnings_dict[latest_date_scrape]['eps_dict']['eps_forecast'][0])
                        if eps_forecast < prev_forecast:
                            color3 = 'red'
                        elif eps_forecast > prev_forecast:
                            color3 = 'green'
                        else:
                            color3 = 'gray'
                    except: pass

                    return  dbc.Row([
                        dbc.Col(f'Earnings Date: {earnings_date}', style = {'border': f'5px outset {color}', 'backgroundColor': f'{color}', 'fontSize': 20}),
                        dbc.Col(f'EPS Suprise ({latest_report_date}): {eps_suprise}%', style = {'border': f'5px outset {color1}', 'backgroundColor': f'{color1}', 'fontSize': 20}),
                        dbc.Col(f'EPS Forecast: {eps_forecast}', style = {'border': f'5px outset {color3}', 'backgroundColor': f'{color3}', 'fontSize': 20}),
                        dbc.Col(f'Forecast Change: {analyst_sentiment_nextq}', style = {'border': f'5px outset {color2}', 'backgroundColor': f'{color2}', 'fontSize': 20}),
                    ])
                else: return dbc.Col(f'Need to Download {ticker} Earnings', style = {'border': f'5px outset gray', 'backgroundColor': f'gray', 'fontSize': 20})
            except:
                return  dbc.Row([
                    dbc.Col(f'Earnings Date: {earnings_date}', style = {'border': f'5px outset {color}', 'backgroundColor': f'{color}', 'fontSize': 20}),
                    dbc.Col(f'EPS Suprise: {eps_suprise}', style = {'border': f'5px outset {color1}', 'backgroundColor': f'{color1}', 'fontSize': 20}),
                    dbc.Col(f'EPS Forecast: {eps_forecast}', style = {'border': f'5px outset {color3}', 'backgroundColor': f'{color3}', 'fontSize': 20}),
                    dbc.Col(f'Forecast Change: {analyst_sentiment_nextq}', style = {'border': f'5px outset {color2}', 'backgroundColor': f'{color2}', 'fontSize': 20}),
                    ])

        @app.callback(Output('short-interest', 'children'), Input('ticker', 'value'))
        def short_interest(ticker):
            today = datetime.today()
            end = today.strftime('%Y-%m-%d')
            start_date = today - timedelta(days=14)
            start = start_date.strftime('%Y-%m-%d')
            df_si = self.finra.get_si(ticker, start, end, redownload = False)
            if df_si.shape[0] > 0:
                short_interest = df_si['ShortInterest'].iloc[-1]
                avg_short_interest = df_si['ShortInterest'].iloc[:-1].mean()
                std_dev = df_si['ShortInterest'].iloc[:-1].std()

                si_dev_from_mean = round((short_interest-avg_short_interest)/std_dev,2)
                if si_dev_from_mean>0:
                    color = 'red'
                elif si_dev_from_mean<0:
                    color = 'green'
                else:
                    color = 'gray'

                df_si.reset_index(drop=True,inplace=True)
                df_si.reset_index(drop=False,inplace=True)
                best_fit, params = curve_fitting.find_best_fit(df_si['index'],df_si['ShortInterest'])
                r_squared, popt, y_fit, t_fit = params
                si_trend = 'none'
                if best_fit == 'polynomial':
                    if popt[0] < 0:
                        si_trend = 'poly down'
                    else:
                        si_trend = 'poly up'
                elif best_fit == 'power':
                    if (popt[0] < 0) or (popt[1] < 1):
                        si_trend = 'power down'
                    else:
                        si_trend = 'power up'
                elif best_fit == 'exponential':
                    if popt[0] < 1:
                        si_trend = 'exp down'
                    else:
                        si_trend = 'exp up'
                elif best_fit == 'linear':
                    if popt[0] > 0:
                        si_trend = 'linear up'
                    else:
                        si_trend = 'linear down'
                if 'up' in si_trend:
                    color1 = 'red'
                elif 'down' in si_trend:
                    color1 = 'green'
                else:
                    color1 = 'gray'

                #make small fig
                fig = make_subplots(rows=1,cols=1)
                fig.add_trace(go.Scatter(x=df_si['index'],y=df_si['ShortInterest']))
                fig.add_trace(go.Scatter(x=t_fit,y=y_fit, mode='lines'))
                fig.update_layout(height=60,width=100, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',showlegend=False, margin={'b':0,'l':0,'r':0,'t':0})
                fig.update_xaxes(rangeslider_visible=False,showline=False,showgrid=False)
                fig.update_yaxes(showline=False,showgrid=False, gridwidth=0.001)

                return  dbc.Row([
                    dbc.Col(f'SI Deviation: {si_dev_from_mean}', style = {'border': f'5px outset {color}', 'backgroundColor': f'{color}', 'fontSize': 20}),
                    dbc.Col(f'SI Trend: {si_trend}', style = {'border': f'5px outset {color1}', 'backgroundColor': f'{color1}', 'fontSize': 20}),
                    dbc.Col(dcc.Graph(figure=fig))
                ])
            else:
                return  dbc.Row([
                    dbc.Col(f'SI Deviation: N/A', style = {'border': f'5px outset gray', 'backgroundColor': f'gray', 'fontSize': 20}),
                    dbc.Col(f'SI Trend: N/A', style = {'border': f'5px outset gray', 'backgroundColor': f'gray', 'fontSize': 20})
                ])

        @app.callback(Output('update-earnings', 'children'), Input('download-earnings', 'n_clicks'), Input('ticker', 'value'))
        def update_earnings(n_clicks, ticker):
            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            if (n_clicks != 0) and ('download-earnings' in changed_id):
                driver = Nasdaq()
                driver.get_earnings(ticker)
            return

        @app.callback(Output('get-candles', 'children'), Input('load-candles', 'n_clicks'), Input('start-time', 'value'), Input('end-time', 'value'), Input('granularity', 'value'), Input('date-range', 'start_date'), Input('date-range', 'end_date'), Input('ticker', 'value'))
        def candles(n_clicks_candles, start_time, end_time, granularity, start_date, end_date, ticker):
            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            fig = make_subplots(rows=2,cols=1, specs=[[{"secondary_y": False}],[{"secondary_y": True}]], shared_xaxes=True)
            if (start_time is not None) and (end_time is not None) and (granularity is not None) and (start_date is not None) and (end_date is not None) and (ticker is not None):
                if (n_clicks_candles%2 == 0) and ('load-candles' in changed_id):
                        try:
                            start = start_date + 'T' + start_time #+ ':00-00:00'
                            end = end_date + 'T' + end_time #+ ':00-00:00'

                            start = parse(start)
                            end = parse(end)

                            start = start.astimezone(pytz.timezone('US/Eastern'))
                            end = end.astimezone(pytz.timezone('US/Eastern'))

                            date,t = str(start).split()
                            start = date + 'T' + t
                            date,t = str(end).split()
                            end = date + 'T' + t

                            code, data = self.api.get_historical_bars(ticker,start,end,granularity)

                            if 'bars' in data.keys():
                                if data['bars'] is not None:
                                    self.candles = pd.DataFrame(data['bars'])
                                    self.candles.t = self.candles.t.apply(lambda t:parse(t).astimezone(pytz.timezone('US/Eastern')))
                                    delta_price = round(100*(data['bars'][-1]['c'] - data['bars'][0]['c'])/data['bars'][0]['c'],2)
                                    if delta_price>0:
                                        color = 'green'
                                    else:
                                        color = 'red'
                                    fig.add_trace(go.Candlestick(
                                        x=self.candles.t,
                                        close=self.candles.c,
                                        open=self.candles.o,
                                        high=self.candles.h,
                                        low=self.candles.l), row=1, col=1)

                                    fig.update_layout(margin={'b':0,'l':0,'r':0,'t':0}, title = {'text': f'{ticker}<br>{delta_price}%','font':{'size':30,'color':color},'y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'})
                                    
                            self.finra.download_finra_data(end_date)
                            df_si = self.finra.get_si(ticker, start_date, end_date, redownload = False)

                            if df_si.shape[0]>0:
                                df_si['Date'] = df_si['Date'].apply(lambda t:str(t)[:-8] + '12:45:00')
                                df_si['LongVolume'] = df_si['TotalVolume'] - df_si['ShortVolume']
                                fig.add_trace(go.Bar(name='Short Volume', x=df_si['Date'], y=df_si['ShortVolume'], xperiod=1000*60*60*16/2,xperiodalignment="end"),secondary_y=False,row=2,col=1)
                                fig.add_trace(go.Bar(name='Long Volume', x=df_si['Date'], y=df_si['LongVolume'], xperiod=1000*60*60*16/2,xperiodalignment="start"),secondary_y=False,row=2,col=1)
                                fig.add_trace(go.Scatter(name='Short Interest', x=df_si['Date'], y=df_si['ShortInterest']),secondary_y=True,row=2,col=1)

                            if granularity[-3:] == 'Day':
                                rangebreaks = [
                                    dict(bounds=["sat", "mon"]),
                                    dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                                ]
                            else:
                                rangebreaks = [
                                    dict(bounds=["sat", "mon"]),
                                    dict(bounds=[21, 3], pattern="hour"),
                                    dict(values=["2019-12-25", "2020-12-24"])  # hide holidays (Christmas and New Year's, etc)
                                ]

                            fig['layout']['yaxis1'].update(domain=[0.1,1])
                            fig['layout']['yaxis2'].update(domain=[0,0.1])

                            fig.update_xaxes(rangeslider_visible=False, rangebreaks=rangebreaks,showline=False,linewidth=0.001,showgrid=False,gridwidth=0.001,rangeslider={'visible': False})
                            fig.update_yaxes(showline=False, linewidth=0.001, showgrid=False, gridwidth=0.001)

                        except Exception as e: print(e)

            fig.update_layout(height=800, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(x=0.8,y=1.2))
            return dcc.Graph(figure=fig)

        @app.callback(Output('save-tickers', 'children'), Input('ticker', 'value'), Input('save-tickers-button', 'n_clicks'), Input('clear-tickers-button', 'n_clicks'))
        def save_ticker(ticker, n_clicks_save, n_clicks_clear):
            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            if (n_clicks_save != 0) and ('save-tickers-button' in changed_id):
                if ticker in self.tradable_tickers:
                    if os.path.isfile('dash_stream/ticker.json') == True:
                        with open('dash_stream/ticker.json') as f:
                            ticker_dict = json.load(f)
                            if ticker in ticker_dict['tickers']:
                                pass
                            else:
                                ticker_dict['tickers'] += [ticker]
                    else:
                        ticker_dict = {'tickers':[ticker]}
                    
                    with open('dash_stream/ticker.json', 'w') as f:
                        json.dump(ticker_dict, f, ensure_ascii=False, indent=4)
                    
                    output = ', '.join(str(ticker) for ticker in ticker_dict['tickers'])
                    return output

            elif (n_clicks_clear != 0) and ('clear-tickers' in changed_id):
                ticker_dict = {'tickers':[]}

                with open('dash_stream/ticker.json', 'w') as f:
                    json.dump(ticker_dict, f, ensure_ascii=False, indent=4)
            return []
     
        @app.callback(Output('stream-quotes', 'children'), Input('ticker', 'value'), Input('start-stream','value'))
        def stream_quotes(ticker, stream):
            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            if ('start-stream' in changed_id):
                if stream == True:
                    if ticker in self.tradable_tickers:
                        self.api.stream_quotes([ticker], reset=True)
                    else:
                        print('ticker', ticker, 'not found')
                else:
                    self.api.close_ws()

        @app.callback(Output('live-stream-quotes', 'children'), Input('start-stream', 'value'), Input('quotes-interval', 'n_intervals'))
        def quote_stream_fig(stream, interval):
            if stream == True:
                quotes = self.load_quotes()

                if quotes is not None:
                    tickers = list(quotes.keys())
                    specs = [[{"secondary_y": True}] for _ in range(len(tickers))]
                    titles = tuple([ticker for ticker in tickers])

                    fig = make_subplots(
                        rows=len(tickers),
                        cols=1,
                        specs=specs,
                        subplot_titles=titles,
                        shared_xaxes=True)
                    
                    for n,ticker in enumerate(tickers):
                        price = quotes[ticker]['price']
                        p_time = quotes[ticker]['time']

                        utc = [parse(datetime.utcfromtimestamp(int(t)).strftime('%Y-%m-%d %H:%M:%S')) for t in p_time]
                        p_time = [t.replace(tzinfo=timezone.utc).astimezone(tz=None) for t in utc]

                        delta = np.diff(price)/price[:-1]*100
                        d_time = p_time[1:]

                        if len(price) > 0:
                            fig.add_trace(go.Scatter(
                                x=p_time,
                                y=price,
                                mode = 'markers+lines',
                                hoverinfo='name',
                                line_color= 'rgb(10, 6, 200)',
                                showlegend= False), row=n+1, col=1, secondary_y=False)

                            d1 = (len(tickers)-1-n)/len(tickers)
                            d2 = (len(tickers)-n)/len(tickers)
                            spacing = 0.01/len(tickers)

                            fig['layout'][f'yaxis{2*n+1}'].update(
                                domain=[d1+spacing,d2-spacing],
                                title={'text':'Price', 'font':{'size':20}})
                            
                        if len(delta)>1:
                            if delta[-1] > 0:
                                line_color = 'rgb(20, 140, 30)'
                            else:
                                line_color = 'rgb(255, 79, 38)'
                                
                            fig.add_trace(go.Scatter(
                                x=d_time,
                                y=delta,
                                marker=dict(size=12),
                                mode = 'markers',
                                hoverinfo='name',
                                line_color= line_color,
                                showlegend= False), row=n+1, col=1, secondary_y=True)
                                
                            d1 = (len(tickers)-1-n)/len(tickers)
                            d2 = (len(tickers)-n)/len(tickers)
                            spacing = 0.01/len(tickers)

                            fig['layout'][f'yaxis{2*n+2}'].update(
                                domain=[d1+spacing,d2-spacing],
                                title={'text':'Delta', 'font':{'size':20}})
                    
                    maxtime = datetime.now()
                    mintime = datetime.now()-timedelta(minutes=3)
                    fig.update_xaxes(
                        range=[mintime,maxtime],
                        mirror=True,
                        rangeslider_visible=False,
                        showline=True,
                        linewidth=0.001,
                        showgrid=False,
                        gridwidth=0.001)

                    fig.update_yaxes(
                        mirror=True,
                        showline=True,
                        linewidth=0.001,
                        showgrid=False,
                        gridwidth=0.001)

                    fig.update_layout(
                        height =len(tickers)*700,
                        # transition=dict(duration=1,easing='elastic'),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)')

                    for n, annotation in enumerate(fig['layout']['annotations']):
                        annotation['y']=(len(tickers)-n)/len(tickers)- 0.15/len(tickers)
                        annotation['font']['size'] = 40

                    return dcc.Graph(figure=fig, animate=False, config={'responsive':True})

            fig = px.scatter(x=[np.nan], y=[np.nan])
            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return dcc.Graph(figure=fig)

        @app.callback(Output('institutional-hist', 'children'), Input('load-hist-button', 'n_clicks'))
        def thirteen_f(n_clicks):
            pass

if __name__ == '__main__':
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)
    dash_inst = Sentiment_Dash()
    dash_inst.build_app(app)
    app.run_server(debug=True)
