# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

# Setting up page configurations
st.set_page_config(
    page_title="Database",
    page_icon="📂",
    layout="wide")

# Removing the arrow from metrics
st.write("""<style>[data-testid="stMetricDelta"] svg {display: none;}</style>""",unsafe_allow_html=True,)

# Expanding the width of the sidebar (to better fit feature names in the sidebar)
#st.markdown("""<style>[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {width: 400px;}[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {width: 400px;}</style>""",unsafe_allow_html=True)

# Loading data only once
@st.experimental_singleton
def read_objects():
    # Importing datasets
    BL18 = pd.read_csv(r'CSV files/Bundesliga18_rating.csv', sep=",", decimal=",").reset_index(drop=True).iloc[1: , :]
    BL19 = pd.read_csv(r'CSV files/Bundesliga19_rating.csv', sep=",", decimal=",").reset_index(drop=True)
    BL20 = pd.read_csv(r'CSV files/Bundesliga20_rating.csv', sep=",", decimal=",").reset_index(drop=True)
    BL21 = pd.read_csv(r'CSV files/Bundesliga21_rating.csv', sep=",", decimal=",").reset_index(drop=True)
    BL22 = pd.read_csv(r'CSV files/Bundesliga22_rating.csv', sep=",", decimal=",").reset_index(drop=True)
    SL21 = pd.read_csv(r'CSV files/Superliga21_rating.csv', sep=",", decimal=",").reset_index(drop=True)
    SL22 = pd.read_csv(r'CSV files/Superliga22_rating.csv', sep=",", decimal=",").reset_index(drop=True)

    BL18 = BL18.dropna(subset=['Duels per 90'])
    BL19 = BL19.dropna(subset=['Duels per 90'])
    BL20 = BL20.dropna(subset=['Duels per 90'])

    # Creating a function to clean the data
    def clean_data_league(df):
        cols = df.columns.drop(['Player', 'Team', 'Team within selected timeframe', 'Position', 'Contract expires', 'Birth country', 'Passport country', 'Foot', 'On loan'])
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        df['Contract expires'] =  pd.to_datetime(df['Contract expires'])
        df[['Player', 'Team', 'Team within selected timeframe', 'Position', 'Contract expires', 'Birth country', 'Passport country', 'Foot', 'On loan']] = df[['Player', 'Team', 'Team within selected timeframe', 'Position', 'Contract expires', 'Birth country', 'Passport country', 'Foot', 'On loan']].fillna('Unknown')
        return df
    
    # Applying the function to the data
    BL18 = clean_data_league(BL18)
    BL19 = clean_data_league(BL19)
    BL20 = clean_data_league(BL20)
    BL21 = clean_data_league(BL21)
    BL22 = clean_data_league(BL22)
    SL21 = clean_data_league(SL21)
    SL22 = clean_data_league(SL22)

    # Adding league and year to each dataset before concatenating for filtering purposes
    BL18['League'] = 'Bundesliga'
    BL18['Year'] = '2018'
    BL19['League'] = 'Bundesliga'
    BL19['Year'] = '2019'
    BL20['League'] = 'Bundesliga'
    BL20['Year'] = '2020'
    BL21['League'] = 'Bundesliga'
    BL21['Year'] = '2021'
    BL22['League'] = 'Bundesliga'
    BL22['Year'] = '2022'
    SL21['League'] = 'Superliga'
    SL21['Year'] = '2021'
    SL22['League'] = 'Superliga'
    SL22['Year'] = '2022'

    database = pd.concat([BL18, BL19, BL20, BL21, BL22, SL21, SL22]).reset_index(drop=True)
    
    return database

database = read_objects()



# Extracting column names for metric selection (dropping categorical values and defeault metrics)
metrics = database.apply(pd.to_numeric, errors='coerce')
metrics = database.drop(['League', 'Year', 'Player', 'Position', 'Team', 'Foot', 'Age', 'Height', 'Weight', 'Birth country', 'Matches played', 'Minutes played', 'Yellow cards', 'Red cards', 'Rating', 'Market value', 'Team within selected timeframe', 'Contract expires', 'Passport country', 'On loan', 'Yellow cards per 90', 'Red cards per 90'], axis=1).columns.values.tolist()


# Setting up the sidebar
with st.sidebar:
    league = database['League'].drop_duplicates()
    league = st.selectbox('League', league)

    year = database.loc[(database['League'] == league)]['Year'].drop_duplicates()
    year = st.selectbox('Year', year)

    player = database.loc[(database['League'] == league) & (database['Year'] == year)]['Player'].drop_duplicates()
    player = st.selectbox('Player', player)

    with st.expander('Expand to select metrics to display'):
        col1, col2 = st.columns(2)
        metric_1 = col1.selectbox('', metrics, key=1)
        metric_2 = col2.selectbox('', metrics[1:], key=2)
        metric_3 = col1.selectbox('', metrics[2:], key=3)
        metric_4 = col2.selectbox('', metrics[3:], key=4)
        metric_5 = col1.selectbox('', metrics[4:], key=5)
        metric_6 = col2.selectbox('', metrics[5:], key=6)
        metric_7 = col1.selectbox('', metrics[6:], key=7)
        metric_8 = col2.selectbox('', metrics[7:], key=8)
        metric_9 = col1.selectbox('', metrics[8:], key=9)
        metric_10 = col2.selectbox('', metrics[9:], key=10)



# Defining player information based on selections in the sidebar
position = database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)]['Position'].values[0]
team = database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)]['Team'].values[0]
foot = database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)]['Foot'].values[0]
age = database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)]['Age'].values[0]
height = database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)]['Height'].values[0]
weight = database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)]['Weight'].values[0]
birth_country = database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)]['Birth country'].values[0]

# Defining player metrics based on selections in the sidebar
player_rating = database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)]['Rating'].values[0]
market_value = database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)]['Market value'].values[0]
matches_played = database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)]['Matches played'].values[0]
minutes_played = database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)]['Minutes played'].values[0]
yellow_cards = database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)]['Yellow cards'].values[0]
red_cards = database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)]['Red cards'].values[0]

# Defining averages
matches_played_avg = database.loc[(database['League'] == league) & (database['Year'] == year)]['Matches played']
minutes_played_avg = database.loc[(database['League'] == league) & (database['Year'] == year)]['Minutes played']
yellow_cards_avg = database.loc[(database['League'] == league) & (database['Year'] == year)]['Yellow cards']
red_cards_avg = database.loc[(database['League'] == league) & (database['Year'] == year)]['Red cards']
player_rating_avg = database.loc[(database['League'] == league) & (database['Year'] == year)]['Rating']
market_value_avg = database.loc[(database['League'] == league) & (database['Year'] == year)]['Market value']


# Setting up the page
col1, col2, col3 = st.columns([2, 0.2, 2.6])

# Adding user input from the sidebar to the default page     
col1.header(player)
col1.info('Position: ' + position)
col1.info('Team:  ' + team)
col1.info('Foot:  ' + foot)
col1.info('Age:  ' + str(age))
col1.info('Height:  ' + str(height))
col1.info('Weight:  ' + str(weight))
col1.info('Birth country:  ' + birth_country)
col2.write('')

with col3:
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    categories = [
        {"name": metric_1, "value": database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)][metric_1].values[0]},
        {"name": metric_2, "value": database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)][metric_2].values[0]},
        {"name": metric_3, "value": database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)][metric_3].values[0]},
        {"name": metric_4, "value": database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)][metric_4].values[0]},
        {"name": metric_5, "value": database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)][metric_5].values[0]},
        {"name": metric_6, "value": database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)][metric_6].values[0]},
        {"name": metric_7, "value": database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)][metric_7].values[0]},
        {"name": metric_8, "value": database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)][metric_8].values[0]},
        {"name": metric_9, "value": database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)][metric_9].values[0]},
        {"name": metric_10, "value": database.loc[(database['League'] == league) & (database['Year'] == year) & (database['Player'] == player)][metric_10].values[0]},]

    subplots = make_subplots(
        rows=len(categories),
        cols=1,
        subplot_titles=[x["name"] for x in categories],
        shared_xaxes=True,
        print_grid=False,
        vertical_spacing=(0.50 / len(categories)))
    _ = subplots['layout'].update(
        width=200,
        plot_bgcolor='rgba(0,0,0,0)')

    for k, x in enumerate(categories):
        subplots.add_trace(dict(
            type='bar',
            orientation='h',
            y=[x["name"]],
            x=[x["value"]],
            text=["{:,.0f}".format(x["value"])],
            hoverinfo='text',
            textposition='auto',
            marker=dict(
                color="#1f77b4")), k+1, 1)
    
    subplots['layout'].update(showlegend=False)
    for x in subplots["layout"]['annotations']:
        x['x'] = 0
        x['xanchor'] = 'left'
        x['align'] = 'left'
        x['font'] = dict(size=12)
    
    for axis in subplots['layout']:
        if axis.startswith('yaxis') or axis.startswith('xaxis'):
            subplots['layout'][axis]['visible'] = False

    subplots['layout']['margin'] = {'l': 0,'r': 0,'t': 20,'b': 1,}
    height_calc = 52 * len(categories)
    height_calc = max([height_calc, 500])
    subplots['layout']['height'] = height_calc
    subplots['layout']['width'] = height_calc
    subplots



# Setting up the rest of the page
col1, col2, col3, col4, col5, col6 = st.columns([0.9, 0.9, 0.9, 0.9, 0.9, 1.5])
col1.metric(label="Matches Played", value=matches_played, delta= 'avg. ' + str(round(matches_played_avg.mean(), 2)), delta_color='off')

col2.metric(label="Minutes Played", value=f"{minutes_played:,}", delta='avg. ' + str(f"{round(minutes_played_avg.mean(), 2):,}"), delta_color='off')

col3.metric(label="Yellow Cards", value=yellow_cards, delta='avg. ' + str(round(yellow_cards_avg.mean(), 2)), delta_color='off')

col4.metric(label="Red Cards", value=red_cards, delta='avg. ' + str(round(red_cards_avg.mean(), 2)), delta_color='off')

col5.metric(label="Player Rating", value=player_rating, delta='avg. ' + str(round(player_rating_avg.mean(), 2)), delta_color='off')

col6.metric(label="Market Value", value=f"{market_value:,}", delta='avg. ' + str(f"{round(market_value_avg.mean(), 2):,}"), delta_color='off')