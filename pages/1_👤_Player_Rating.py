# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import shap
#from streamlit_shap import st_shap
import pickle
import plotly.graph_objects as go
fig = go.Figure()
fig.update_layout(margin=dict(l=25, r=25, t=25, b=25), polar=dict(radialaxis=dict(visible=True, range=[0, 100])),showlegend=False)

# Setting up page configurations
st.set_page_config(
    page_title="Player Rating",
    page_icon="👤",
    layout="wide")

# Expanding the width of the sidebar (to fit feature names in one line in the sidebar)
#st.markdown("""<style>[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {width: 500px;}[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {width: 500px;}</style>""",unsafe_allow_html=True)

# Loading data, models, scalers, explainers, etc., only once
@st.experimental_singleton
def read_objects():
    # Importing datasets
    BL18 = pd.read_csv(r'CSV files/Bundesliga18_rating.csv', sep=",", decimal=",").reset_index(drop=True)
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

    BL = pd.concat([BL18, BL19, BL20, BL21, BL22, SL21, SL22]).reset_index(drop=True)
    return BL

BL = read_objects()


# Setting up the default sidebar
with st.sidebar:
    with st.expander('Expand to fill in player information'):
        col1, col2 = st.columns(2)
        player = col1.text_input((''),('Player'))
        team = col1.text_input((''),('Team'))
        age = col1.text_input((''),('Age'))
        weight = col1.text_input((''),('Weight'))

        position = col2.selectbox((''),('Position', 'Goalkeeper', 'Central Defender', 'Full Back', 'Defensive Midfielder', 'Central Midfielder', 'Attacking Midfielder', 'Winger Midfielder', 'Forwarder'))
        foot = col2.selectbox((''),('Foot', 'Right', 'Left', 'Both', 'Unknown'))
        height = col2.text_input((''),('Height'))
        nationality = col2.text_input((''),('Nationality'))



# Setting up the default page
if position == 'Position':

    # Setting up the default page
    col1, col2 = st.columns([2, 3])

    # Adding user input from the sidebar to the default page     
    if player == 'Player': col1.header('Player')
    else: col1.header(player)
    if position == 'Position': col1.info('Position:  ')
    else: col1.info('Position:  ' + position)
    if team == 'Team': col1.info('Team:  ')
    else: col1.info('Team:  ' + team)
    if foot == 'Foot': col1.info('Foot:  ')
    else: col1.info('Foot:  ' + foot)
    if age == 'Age': col1.info('Age:  ')
    else: col1.info('Age:  ' + age)
    if height == 'Height': col1.info('Height:  ')
    else: col1.info('Height:  ' + height)
    if weight == 'Weight': col1.info('Weight:  ')
    else: col1.info('Weight:  ' + weight)

    # Setting up the default radar graph
    col2.write('')
    col2.write('')
    categories = ['Feature 0', 'Feature 1', 'Feature 2','Feature 3','Feature 4','Feature 5','Feature 6','Feature 7','Feature 8', 'Feature 9']
    fig.add_trace(go.Scatterpolar(r=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], theta=categories, fill='toself', name=player))
    col2.plotly_chart(fig, use_container_width=True)

    # Setting up the rest of the default page
    col1, col2, col3, col4, col5, col6= st.columns([2, 0.8, 1, 0.1, 1, 0.1])

    # Adding user input from the sidebar to the default page
    if nationality == 'Nationality': col1.info('Nationality:  ')
    else: col1.info('Nationality:  ' + nationality)

    # Setting up the rest of the default page
    col3.metric(label="Player Rating", value='00')
    col4.write('')
    col5.metric(label="Mean Error", value='00.0')
    col6.write('')
    with st.expander("Player Rating Explained"):
        st.write('')












# Position GK: Goalkeeper

# Loading models, scalers, explainers, etc., only once
@st.experimental_singleton
def read_objects():
    # Model, scaler, explainer and features selected for each position
    GK_model = pickle.load(open('Pickles/1_GK/GK_model.pkl','rb'))
    GK_scaler = pickle.load(open('Pickles/1_GK/GK_scaler.pkl','rb'))
    #GK_shap_values = pickle.load(open('Pickles/1_GK/GK_shap.pkl','rb'))
    GK_rmse  = pickle.load(open('Pickles/1_GK/GK_rmse.pkl','rb'))
    #GK_explainer = shap.TreeExplainer(GK_model)
    GK_fs = pd.read_csv('Pickles/1_GK/GK_fs.csv')

    return GK_model, GK_scaler, GK_rmse, GK_explainer, GK_fs

GK_model, GK_scaler, GK_rmse, GK_explainer, GK_fs = read_objects()

# Setting up the page for position GK
if position == 'Goalkeeper':

    # Setting up the sidebar
    with st.sidebar:
        with st.expander('Expand to fill in player performance'):
            col1, col2 = st.columns(2)
            feature_0 = col1.number_input((GK_fs.iloc[0].values[0]), key=0, min_value=0, step=10)
            feature_2 = col1.number_input((GK_fs.iloc[2].values[0]), key=2, min_value=0, step=200)
            feature_4 = col1.number_input((GK_fs.iloc[4].values[0]), key=4, min_value=0, step=10)
            feature_6 = col1.number_input((GK_fs.iloc[6].values[0]), key=6, min_value=0, step=2)
            feature_8 = col1.number_input((GK_fs.iloc[8].values[0]), key=8, min_value=0, step=10)
            
            feature_1 = col2.number_input((GK_fs.iloc[1].values[0]), key=1, min_value=0, step=2)
            feature_3 = col2.number_input((GK_fs.iloc[3].values[0]), key=3, min_value=0, step=10)
            feature_5 = col2.number_input((GK_fs.iloc[5].values[0]), key=5, min_value=0, step=10)
            feature_7 = col2.number_input((GK_fs.iloc[7].values[0]), key=7, min_value=0, step=20)
            feature_9 = col2.number_input((GK_fs.iloc[9].values[0]), key=9, min_value=0, step=5)
            
            # Adding a button that triggers prediction of the rating
            predict_button = st.button('Predict Player Rating')
            compare_BL = st.checkbox('Compare ' + player + ' to the highest rated ' + position + ' in the Bundesliga')

    # Setting up the page
    col1, col2 = st.columns([2, 3])

    # Adding user input from the sidebar to the page
    if player == 'Player': col1.header('Player')
    else: col1.header(player)
    if position == 'Position': col1.info('Position:  ')
    else: col1.info('Position:  ' + position)
    if team == 'Team': col1.info('Team:  ')
    else: col1.info('Team:  ' + team)
    if foot == 'Foot': col1.info('Foot:  ')
    else: col1.info('Foot:  ' + foot)
    if age == 'Age': col1.info('Age:  ')
    else: col1.info('Age:  ' + age)
    if height == 'Height': col1.info('Height:  ')
    else: col1.info('Height:  ' + height)
    if weight == 'Weight': col1.info('Weight:  ')
    else: col1.info('Weight:  ' + weight)

    # Setting up the radar graph
    col2.write('')
    col2.write('')
    categories =    [GK_fs.iloc[0],
                    GK_fs.iloc[1],
                    GK_fs.iloc[2],
                    GK_fs.iloc[3],
                    GK_fs.iloc[4],
                    GK_fs.iloc[5],
                    GK_fs.iloc[6],
                    GK_fs.iloc[7],
                    GK_fs.iloc[8],
                    GK_fs.iloc[9]]
    fig.add_trace(go.Scatterpolar(r=    [(feature_0 - 0) / (BL[GK_fs.iloc[0].values[0]].max() - 0) * 100,
                                        (feature_1 - 0) / (BL[GK_fs.iloc[1].values[0]].max() - 0) * 100,
                                        (feature_2 - 0) / (BL[GK_fs.iloc[2].values[0]].max() - 0) * 100,
                                        (feature_3 - 0) / (BL[GK_fs.iloc[3].values[0]].max() - 0) * 100,
                                        (feature_4 - 0) / (BL[GK_fs.iloc[4].values[0]].max() - 0) * 100,
                                        (feature_5 - 0) / (BL[GK_fs.iloc[5].values[0]].max() - 0) * 100,
                                        (feature_6 - 0) / (BL[GK_fs.iloc[6].values[0]].max() - 0) * 100,
                                        (feature_7 - 0) / (BL[GK_fs.iloc[7].values[0]].max() - 0) * 100,
                                        (feature_8 - 0) / (BL[GK_fs.iloc[8].values[0]].max() - 0) * 100,
                                        (feature_9 - 0) / (BL[GK_fs.iloc[9].values[0]].max() - 0) * 100
                                        ],theta=categories, fill='toself', name=player))

    if compare_BL:
        feature_0_BL = BL[GK_fs.iloc[0].values[0]].loc[(BL['Position'] == 'GK') & (BL['Rating'] == BL['Rating'].max())].values[0]
        feature_1_BL = BL[GK_fs.iloc[1].values[0]].loc[(BL['Position'] == 'GK') & (BL['Rating'] == BL['Rating'].max())].values[0]
        feature_2_BL = BL[GK_fs.iloc[2].values[0]].loc[(BL['Position'] == 'GK') & (BL['Rating'] == BL['Rating'].max())].values[0]
        feature_3_BL = BL[GK_fs.iloc[3].values[0]].loc[(BL['Position'] == 'GK') & (BL['Rating'] == BL['Rating'].max())].values[0]
        feature_4_BL = BL[GK_fs.iloc[4].values[0]].loc[(BL['Position'] == 'GK') & (BL['Rating'] == BL['Rating'].max())].values[0]
        feature_5_BL = BL[GK_fs.iloc[5].values[0]].loc[(BL['Position'] == 'GK') & (BL['Rating'] == BL['Rating'].max())].values[0]
        feature_6_BL = BL[GK_fs.iloc[6].values[0]].loc[(BL['Position'] == 'GK') & (BL['Rating'] == BL['Rating'].max())].values[0]
        feature_7_BL = BL[GK_fs.iloc[7].values[0]].loc[(BL['Position'] == 'GK') & (BL['Rating'] == BL['Rating'].max())].values[0]
        feature_8_BL = BL[GK_fs.iloc[8].values[0]].loc[(BL['Position'] == 'GK') & (BL['Rating'] == BL['Rating'].max())].values[0]
        feature_9_BL = BL[GK_fs.iloc[9].values[0]].loc[(BL['Position'] == 'GK') & (BL['Rating'] == BL['Rating'].max())].values[0]

        fig.add_trace(go.Scatterpolar(r=    [(feature_0_BL - 0) / (BL[GK_fs.iloc[0].values[0]].max() - 0) * 100,
                                            (feature_1_BL - 0) / (BL[GK_fs.iloc[1].values[0]].max() - 0) * 100,
                                            (feature_2_BL - 0) / (BL[GK_fs.iloc[2].values[0]].max() - 0) * 100, 
                                            (feature_3_BL - 0) / (BL[GK_fs.iloc[3].values[0]].max() - 0) * 100, 
                                            (feature_4_BL - 0) / (BL[GK_fs.iloc[4].values[0]].max() - 0) * 100, 
                                            (feature_5_BL - 0) / (BL[GK_fs.iloc[5].values[0]].max() - 0) * 100, 
                                            (feature_6_BL - 0) / (BL[GK_fs.iloc[6].values[0]].max() - 0) * 100, 
                                            (feature_7_BL - 0) / (BL[GK_fs.iloc[7].values[0]].max() - 0) * 100, 
                                            (feature_8_BL - 0) / (BL[GK_fs.iloc[8].values[0]].max() - 0) * 100, 
                                            (feature_9_BL - 0) / (BL[GK_fs.iloc[9].values[0]].max() - 0) * 100,
                                            ], theta=categories, fill='toself', name=BL.loc[(BL['Position'] == 'GK') & (BL['Rating'] == BL['Rating'].max()), 'Player'].values[0]))

    # Plotting the graph
    col2.plotly_chart(fig, use_container_width=True)

    # Setting up the rest of the page
    col1, col2, col3, col4, col5, col6= st.columns([2, 0.8, 1, 0.1, 1, 0.1])

    # Adding user input from the sidebar to the page
    if nationality == 'Nationality': col1.info('Nationality:  ')
    else: col1.info('Nationality:  ' + nationality)
    col2.write('')

    # Creating an if statement that triggers the prediction when pressing the predict_button
    if predict_button:

        # Creating a dataframe with feature names and user input from the sidebar
        user_input = pd.DataFrame({ GK_fs.iloc[0].values[0]:feature_0,
                                    GK_fs.iloc[1].values[0]:feature_1, 
                                    GK_fs.iloc[2].values[0]:feature_2, 
                                    GK_fs.iloc[3].values[0]:feature_3, 
                                    GK_fs.iloc[4].values[0]:feature_4, 
                                    GK_fs.iloc[5].values[0]:feature_5,
                                    GK_fs.iloc[6].values[0]:feature_6,                                         
                                    GK_fs.iloc[7].values[0]:feature_7, 
                                    GK_fs.iloc[8].values[0]:feature_8, 
                                    GK_fs.iloc[9].values[0]:feature_9, 
                                    }, index=[0]) 
        # Scaling the user input
        user_input_scaled = pd.DataFrame(GK_scaler.transform(user_input), columns = user_input.columns, index=[0])  

        # Predicting the rating based on the user input
        predicted_rating = GK_model.predict(user_input_scaled)
    
        # Displaying the rating and RMSE
        col3.metric(label="Player Rating", value=int(predicted_rating))
        col4.write('')
        col5.metric(label="Mean Error", value=np.round(GK_rmse, decimals = 2))
        col6.write('')

        # Displaying the SHAP values
        #with st.expander("Player Rating Explained"):
        #    shap_value = GK_explainer.shap_values(user_input_scaled)
        #    st_shap(shap.force_plot(GK_explainer.expected_value, shap_value, user_input_scaled), height=150, width=700)
    
    # Default display when the button has not been pressed
    else:
        col3.metric(label="Player Rating", value='00')
        col4.write('')
        col5.metric(label="Mean Error", value='00.0')
        col6.write('')
        with st.expander("Player Rating Explained"):
            st.write('')















# Position CD: Central Defender

# Loading models, scalers, explainers, etc., only once
@st.experimental_singleton
def read_objects():
    # Model, scaler, explainer and features selected for each position
    CD_model = pickle.load(open('Pickles/2_CD/CD_model.pkl','rb'))
    CD_scaler = pickle.load(open('Pickles/2_CD/CD_scaler.pkl','rb'))
    #CD_shap_values = pickle.load(open('Pickles/2_CD/CD_shap.pkl','rb'))
    CD_rmse  = pickle.load(open('Pickles/2_CD/CD_rmse.pkl','rb'))
    #CD_explainer = shap.TreeExplainer(CD_model)
    CD_fs = pd.read_csv('Pickles/2_CD/CD_fs.csv')

    return CD_model, CD_scaler, CD_rmse, CD_explainer, CD_fs

CD_model, CD_scaler, CD_rmse, CD_explainer, CD_fs = read_objects()

# Setting up the page for position CD
if position == 'Central Defender':

    # Setting up the sidebar
    with st.sidebar:
        with st.expander('Expand to fill in player performance'):
            col1, col2 = st.columns(2)
            feature_0 = col1.number_input((CD_fs.iloc[0].values[0]), key=0, min_value=0, step=10)
            feature_2 = col1.number_input((CD_fs.iloc[2].values[0]), key=2, min_value=0, step=10)
            feature_4 = col1.number_input((CD_fs.iloc[4].values[0]), key=4, min_value=0, step=200)
            feature_6 = col1.number_input((CD_fs.iloc[6].values[0]), key=6, min_value=0, step=5)
            feature_8 = col1.number_input((CD_fs.iloc[8].values[0]), key=8, min_value=0, step=5)
            
            feature_1 = col2.number_input((CD_fs.iloc[1].values[0]), key=1, min_value=0, step=10)
            feature_3 = col2.number_input((CD_fs.iloc[3].values[0]), key=3, min_value=0, step=10)
            feature_5 = col2.number_input((CD_fs.iloc[5].values[0]), key=5, min_value=0, step=2)
            feature_7 = col2.number_input((CD_fs.iloc[7].values[0]), key=7, min_value=0, step=10)
            feature_9 = col2.number_input((CD_fs.iloc[9].values[0]), key=9, min_value=0, step=2)
            
            # Adding a button that triggers prediction of the rating
            predict_button = st.button('Predict Player Rating')

    # Setting up the page
    col1, col2 = st.columns([2, 3])

    # Adding user input from the sidebar to the page
    if player == 'Player': col1.header('Player')
    else: col1.header(player)
    if position == 'Position': col1.info('Position:  ')
    else: col1.info('Position:  ' + position)
    if team == 'Team': col1.info('Team:  ')
    else: col1.info('Team:  ' + team)
    if foot == 'Foot': col1.info('Foot:  ')
    else: col1.info('Foot:  ' + foot)
    if age == 'Age': col1.info('Age:  ')
    else: col1.info('Age:  ' + age)
    if height == 'Height': col1.info('Height:  ')
    else: col1.info('Height:  ' + height)
    if weight == 'Weight': col1.info('Weight:  ')
    else: col1.info('Weight:  ' + weight)

    # Setting up the radar graph
    col2.write('')
    col2.write('')
    categories =    [CD_fs.iloc[0],
                    CD_fs.iloc[1],
                    CD_fs.iloc[2],
                    CD_fs.iloc[3],
                    CD_fs.iloc[4],
                    CD_fs.iloc[5],
                    CD_fs.iloc[6],
                    CD_fs.iloc[7],
                    CD_fs.iloc[8],
                    CD_fs.iloc[9]]
    fig.add_trace(go.Scatterpolar(r=    [(feature_0 - 0) / (BL[CD_fs.iloc[0].values[0]].max() - 0) * 100,
                                        (feature_1 - 0) / (BL[CD_fs.iloc[1].values[0]].max() - 0) * 100,
                                        (feature_2 - 0) / (BL[CD_fs.iloc[2].values[0]].max() - 0) * 100,
                                        (feature_3 - 0) / (BL[CD_fs.iloc[3].values[0]].max() - 0) * 100,
                                        (feature_4 - 0) / (BL[CD_fs.iloc[4].values[0]].max() - 0) * 100,
                                        (feature_5 - 0) / (BL[CD_fs.iloc[5].values[0]].max() - 0) * 100,
                                        (feature_6 - 0) / (BL[CD_fs.iloc[6].values[0]].max() - 0) * 100,
                                        (feature_7 - 0) / (BL[CD_fs.iloc[7].values[0]].max() - 0) * 100,
                                        (feature_8 - 0) / (BL[CD_fs.iloc[8].values[0]].max() - 0) * 100,
                                        (feature_9 - 0) / (BL[CD_fs.iloc[9].values[0]].max() - 0) * 100
                                        ],theta=categories, fill='toself', name=player))

    # Plotting the graph
    col2.plotly_chart(fig, use_container_width=True)

    # Setting up the rest of the page
    col1, col2, col3, col4, col5, col6= st.columns([2, 0.8, 1, 0.1, 1, 0.1])

    # Adding user input from the sidebar to the page
    if nationality == 'Nationality': col1.info('Nationality:  ')
    else: col1.info('Nationality:  ' + nationality)
    col2.write('')

    # Creating an if statement that triggers the prediction when pressing the predict_button
    if predict_button:

        # Creating a dataframe with feature names and user input from the sidebar
        user_input = pd.DataFrame({ CD_fs.iloc[0].values[0]:feature_0,
                                    CD_fs.iloc[1].values[0]:feature_1, 
                                    CD_fs.iloc[2].values[0]:feature_2, 
                                    CD_fs.iloc[3].values[0]:feature_3, 
                                    CD_fs.iloc[4].values[0]:feature_4, 
                                    CD_fs.iloc[5].values[0]:feature_5,
                                    CD_fs.iloc[6].values[0]:feature_6,                                         
                                    CD_fs.iloc[7].values[0]:feature_7, 
                                    CD_fs.iloc[8].values[0]:feature_8, 
                                    CD_fs.iloc[9].values[0]:feature_9, 
                                    }, index=[0]) 
        # Scaling the user input
        user_input_scaled = pd.DataFrame(CD_scaler.transform(user_input), columns = user_input.columns, index=[0])  

        # Predicting the rating based on the user input
        predicted_rating = CD_model.predict(user_input_scaled)
    
        # Displaying the rating and RMSE
        col3.metric(label="Player Rating", value=int(predicted_rating))
        col4.write('')
        col5.metric(label="Mean Error", value=np.round(CD_rmse, decimals = 2))
        col6.write('')

        # Displaying the SHAP values
        #with st.expander("Player Rating Explained"):
        #    shap_value = CD_explainer.shap_values(user_input_scaled)
        #    st_shap(shap.force_plot(CD_explainer.expected_value, shap_value, user_input_scaled), height=150, width=700)
    
    # Default display when the button has not been pressed
    else:
        col3.metric(label="Player Rating", value='00')
        col4.write('')
        col5.metric(label="Mean Error", value='00.0')
        col6.write('')
        with st.expander("Player Rating Explained"):
            st.write('')

















# Position FB: Full Back

# Loading models, scalers, explainers, etc., only once
@st.experimental_singleton
def read_objects():
    # Model, scaler, explainer and features selected for each position
    FB_model = pickle.load(open('Pickles/3_FB/FB_model.pkl','rb'))
    FB_scaler = pickle.load(open('Pickles/3_FB/FB_scaler.pkl','rb'))
    #FB_shap_values = pickle.load(open('Pickles/3_FB/FB_shap.pkl','rb'))
    FB_rmse  = pickle.load(open('Pickles/3_FB/FB_rmse.pkl','rb'))
    #FB_explainer = shap.TreeExplainer(FB_model)
    FB_fs = pd.read_csv('Pickles/3_FB/FB_fs.csv')

    return FB_model, FB_scaler, FB_rmse, FB_explainer, FB_fs

FB_model, FB_scaler, FB_rmse, FB_explainer, FB_fs = read_objects()

# Setting up the page for position FB
if position == 'Full Back':

    # Setting up the sidebar
    with st.sidebar:
        with st.expander('Expand to fill in player performance'):
            col1, col2 = st.columns(2)
            feature_0 = col1.number_input((FB_fs.iloc[0].values[0]), key=0, min_value=0, step=10)
            feature_2 = col1.number_input((FB_fs.iloc[2].values[0]), key=2, min_value=0, step=10)
            feature_4 = col1.number_input((FB_fs.iloc[4].values[0]), key=4, min_value=0, step=10)
            feature_6 = col1.number_input((FB_fs.iloc[6].values[0]), key=6, min_value=0, step=1)
            feature_8 = col1.number_input((FB_fs.iloc[8].values[0]), key=8, min_value=0, step=5)
            
            feature_1 = col2.number_input((FB_fs.iloc[1].values[0]), key=1, min_value=0, step=2)
            feature_3 = col2.number_input((FB_fs.iloc[3].values[0]), key=3, min_value=0, step=10)
            feature_5 = col2.number_input((FB_fs.iloc[5].values[0]), key=5, min_value=0, step=200)
            feature_7 = col2.number_input((FB_fs.iloc[7].values[0]), key=7, min_value=0, step=10)
            feature_9 = col2.number_input((FB_fs.iloc[9].values[0]), key=9, min_value=0, step=10)
            
            # Adding a button that triggers prediction of the rating
            predict_button = st.button('Predict Player Rating')

    # Setting up the page
    col1, col2 = st.columns([2, 3])

    # Adding user input from the sidebar to the page
    if player == 'Player': col1.header('Player')
    else: col1.header(player)
    if position == 'Position': col1.info('Position:  ')
    else: col1.info('Position:  ' + position)
    if team == 'Team': col1.info('Team:  ')
    else: col1.info('Team:  ' + team)
    if foot == 'Foot': col1.info('Foot:  ')
    else: col1.info('Foot:  ' + foot)
    if age == 'Age': col1.info('Age:  ')
    else: col1.info('Age:  ' + age)
    if height == 'Height': col1.info('Height:  ')
    else: col1.info('Height:  ' + height)
    if weight == 'Weight': col1.info('Weight:  ')
    else: col1.info('Weight:  ' + weight)

    # Setting up the radar graph
    col2.write('')
    col2.write('')
    categories =    [FB_fs.iloc[0],
                    FB_fs.iloc[1],
                    FB_fs.iloc[2],
                    FB_fs.iloc[3],
                    FB_fs.iloc[4],
                    FB_fs.iloc[5],
                    FB_fs.iloc[6],
                    FB_fs.iloc[7],
                    FB_fs.iloc[8],
                    FB_fs.iloc[9]]
    fig.add_trace(go.Scatterpolar(r=    [(feature_0 - 0) / (BL[FB_fs.iloc[0].values[0]].max() - 0) * 100,
                                        (feature_1 - 0) / (BL[FB_fs.iloc[1].values[0]].max() - 0) * 100,
                                        (feature_2 - 0) / (BL[FB_fs.iloc[2].values[0]].max() - 0) * 100,
                                        (feature_3 - 0) / (BL[FB_fs.iloc[3].values[0]].max() - 0) * 100,
                                        (feature_4 - 0) / (BL[FB_fs.iloc[4].values[0]].max() - 0) * 100,
                                        (feature_5 - 0) / (BL[FB_fs.iloc[5].values[0]].max() - 0) * 100,
                                        (feature_6 - 0) / (BL[FB_fs.iloc[6].values[0]].max() - 0) * 100,
                                        (feature_7 - 0) / (BL[FB_fs.iloc[7].values[0]].max() - 0) * 100,
                                        (feature_8 - 0) / (BL[FB_fs.iloc[8].values[0]].max() - 0) * 100,
                                        (feature_9 - 0) / (BL[FB_fs.iloc[9].values[0]].max() - 0) * 100
                                        ],theta=categories, fill='toself', name=player))

    # Plotting the graph
    col2.plotly_chart(fig, use_container_width=True)

    # Setting up the rest of the page
    col1, col2, col3, col4, col5, col6= st.columns([2, 0.8, 1, 0.1, 1, 0.1])

    # Adding user input from the sidebar to the page
    if nationality == 'Nationality': col1.info('Nationality:  ')
    else: col1.info('Nationality:  ' + nationality)
    col2.write('')

    # Creating an if statement that triggers the prediction when pressing the predict_button
    if predict_button:

        # Creating a dataframe with feature names and user input from the sidebar
        user_input = pd.DataFrame({ FB_fs.iloc[0].values[0]:feature_0,
                                    FB_fs.iloc[1].values[0]:feature_1, 
                                    FB_fs.iloc[2].values[0]:feature_2, 
                                    FB_fs.iloc[3].values[0]:feature_3, 
                                    FB_fs.iloc[4].values[0]:feature_4, 
                                    FB_fs.iloc[5].values[0]:feature_5,
                                    FB_fs.iloc[6].values[0]:feature_6,                                         
                                    FB_fs.iloc[7].values[0]:feature_7, 
                                    FB_fs.iloc[8].values[0]:feature_8, 
                                    FB_fs.iloc[9].values[0]:feature_9, 
                                    }, index=[0]) 
        # Scaling the user input
        user_input_scaled = pd.DataFrame(FB_scaler.transform(user_input), columns = user_input.columns, index=[0])  

        # Predicting the rating based on the user input
        predicted_rating = FB_model.predict(user_input_scaled)
    
        # Displaying the rating and RMSE
        col3.metric(label="Player Rating", value=int(predicted_rating))
        col4.write('')
        col5.metric(label="Mean Error", value=np.round(FB_rmse, decimals = 2))
        col6.write('')

        # Displaying the SHAP values
        #with st.expander("Player Rating Explained"):
        #    shap_value = FB_explainer.shap_values(user_input_scaled)
        #    st_shap(shap.force_plot(FB_explainer.expected_value, shap_value, user_input_scaled), height=150, width=700)
    
    # Default display when the button has not been pressed
    else:
        col3.metric(label="Player Rating", value='00')
        col4.write('')
        col5.metric(label="Mean Error", value='00.0')
        col6.write('')
        with st.expander("Player Rating Explained"):
            st.write('')















# Position DMF: Defensive Midfielder

# Loading models, scalers, explainers, etc., only once
@st.experimental_singleton
def read_objects():
    # Model, scaler, explainer and features selected for each position
    DMF_model = pickle.load(open('Pickles/4_DMF/DMF_model.pkl','rb'))
    DMF_scaler = pickle.load(open('Pickles/4_DMF/DMF_scaler.pkl','rb'))
    #DMF_shap_values = pickle.load(open('Pickles/4_DMF/DMF_shap.pkl','rb'))
    DMF_rmse  = pickle.load(open('Pickles/4_DMF/DMF_rmse.pkl','rb'))
    #DMF_explainer = shap.TreeExplainer(DMF_model)
    DMF_fs = pd.read_csv('Pickles/4_DMF/DMF_fs.csv')

    return DMF_model, DMF_scaler, DMF_rmse, DMF_explainer, DMF_fs

DMF_model, DMF_scaler, DMF_rmse, DMF_explainer, DMF_fs = read_objects()

# Setting up the page for position DMF
if position == 'Defensive Midfielder':

    # Setting up the sidebar
    with st.sidebar:
        with st.expander('Expand to fill in player performance'):
            col1, col2 = st.columns(2)
            feature_0 = col1.number_input((DMF_fs.iloc[0].values[0]), key=0, min_value=0, step=10)
            feature_2 = col1.number_input((DMF_fs.iloc[2].values[0]), key=2, min_value=0, step=10)
            feature_4 = col1.number_input((DMF_fs.iloc[4].values[0]), key=4, min_value=0, step=2)
            feature_6 = col1.number_input((DMF_fs.iloc[6].values[0]), key=6, min_value=0, step=2)
            feature_8 = col1.number_input((DMF_fs.iloc[8].values[0]), key=8, min_value=0, step=2)
            
            feature_1 = col2.number_input((DMF_fs.iloc[1].values[0]), key=1, min_value=0, step=10)
            feature_3 = col2.number_input((DMF_fs.iloc[3].values[0]), key=3, min_value=0, step=1)
            feature_5 = col2.number_input((DMF_fs.iloc[5].values[0]), key=5, min_value=0, step=1)
            feature_7 = col2.number_input((DMF_fs.iloc[7].values[0]), key=7, min_value=0, step=10)
            feature_9 = col2.number_input((DMF_fs.iloc[9].values[0]), key=9, min_value=0, step=5)
            
            # Adding a button that triggers prediction of the rating
            predict_button = st.button('Predict Player Rating')

    # Setting up the page
    col1, col2 = st.columns([2, 3])

    # Adding user input from the sidebar to the page
    if player == 'Player': col1.header('Player')
    else: col1.header(player)
    if position == 'Position': col1.info('Position:  ')
    else: col1.info('Position:  ' + position)
    if team == 'Team': col1.info('Team:  ')
    else: col1.info('Team:  ' + team)
    if foot == 'Foot': col1.info('Foot:  ')
    else: col1.info('Foot:  ' + foot)
    if age == 'Age': col1.info('Age:  ')
    else: col1.info('Age:  ' + age)
    if height == 'Height': col1.info('Height:  ')
    else: col1.info('Height:  ' + height)
    if weight == 'Weight': col1.info('Weight:  ')
    else: col1.info('Weight:  ' + weight)

    # Setting up the radar graph
    col2.write('')
    col2.write('')
    categories =    [DMF_fs.iloc[0],
                    DMF_fs.iloc[1],
                    DMF_fs.iloc[2],
                    DMF_fs.iloc[3],
                    DMF_fs.iloc[4],
                    DMF_fs.iloc[5],
                    DMF_fs.iloc[6],
                    DMF_fs.iloc[7],
                    DMF_fs.iloc[8],
                    DMF_fs.iloc[9]]
    fig.add_trace(go.Scatterpolar(r=    [(feature_0 - 0) / (BL[DMF_fs.iloc[0].values[0]].max() - 0) * 100,
                                        (feature_1 - 0) / (BL[DMF_fs.iloc[1].values[0]].max() - 0) * 100,
                                        (feature_2 - 0) / (BL[DMF_fs.iloc[2].values[0]].max() - 0) * 100,
                                        (feature_3 - 0) / (BL[DMF_fs.iloc[3].values[0]].max() - 0) * 100,
                                        (feature_4 - 0) / (BL[DMF_fs.iloc[4].values[0]].max() - 0) * 100,
                                        (feature_5 - 0) / (BL[DMF_fs.iloc[5].values[0]].max() - 0) * 100,
                                        (feature_6 - 0) / (BL[DMF_fs.iloc[6].values[0]].max() - 0) * 100,
                                        (feature_7 - 0) / (BL[DMF_fs.iloc[7].values[0]].max() - 0) * 100,
                                        (feature_8 - 0) / (BL[DMF_fs.iloc[8].values[0]].max() - 0) * 100,
                                        (feature_9 - 0) / (BL[DMF_fs.iloc[9].values[0]].max() - 0) * 100
                                        ],theta=categories, fill='toself', name=player))

    # Plotting the graph
    col2.plotly_chart(fig, use_container_width=True)

    # Setting up the rest of the page
    col1, col2, col3, col4, col5, col6= st.columns([2, 0.8, 1, 0.1, 1, 0.1])

    # Adding user input from the sidebar to the page
    if nationality == 'Nationality': col1.info('Nationality:  ')
    else: col1.info('Nationality:  ' + nationality)
    col2.write('')

    # Creating an if statement that triggers the prediction when pressing the predict_button
    if predict_button:

        # Creating a dataframe with feature names and user input from the sidebar
        user_input = pd.DataFrame({ DMF_fs.iloc[0].values[0]:feature_0,
                                    DMF_fs.iloc[1].values[0]:feature_1, 
                                    DMF_fs.iloc[2].values[0]:feature_2, 
                                    DMF_fs.iloc[3].values[0]:feature_3, 
                                    DMF_fs.iloc[4].values[0]:feature_4, 
                                    DMF_fs.iloc[5].values[0]:feature_5,
                                    DMF_fs.iloc[6].values[0]:feature_6,                                         
                                    DMF_fs.iloc[7].values[0]:feature_7, 
                                    DMF_fs.iloc[8].values[0]:feature_8, 
                                    DMF_fs.iloc[9].values[0]:feature_9, 
                                    }, index=[0]) 
        # Scaling the user input
        user_input_scaled = pd.DataFrame(DMF_scaler.transform(user_input), columns = user_input.columns, index=[0])  

        # Predicting the rating based on the user input
        predicted_rating = DMF_model.predict(user_input_scaled)
    
        # Displaying the rating and RMSE
        col3.metric(label="Player Rating", value=int(predicted_rating))
        col4.write('')
        col5.metric(label="Mean Error", value=np.round(DMF_rmse, decimals = 2))
        col6.write('')

        # Displaying the SHAP values
        #with st.expander("Player Rating Explained"):
        #    shap_value = DMF_explainer.shap_values(user_input_scaled)
        #    st_shap(shap.force_plot(DMF_explainer.expected_value, shap_value, user_input_scaled), height=150, width=700)
    
    # Default display when the button has not been pressed
    else:
        col3.metric(label="Player Rating", value='00')
        col4.write('')
        col5.metric(label="Mean Error", value='00.0')
        col6.write('')
        with st.expander("Player Rating Explained"):
            st.write('')















# Position CMF: Central Midfielder

# Loading models, scalers, explainers, etc., only once
@st.experimental_singleton
def read_objects():
    # Model, scaler, explainer and features selected for each position
    CMF_model = pickle.load(open('Pickles/5_CMF/CMF_model.pkl','rb'))
    CMF_scaler = pickle.load(open('Pickles/5_CMF/CMF_scaler.pkl','rb'))
    #CMF_shap_values = pickle.load(open('Pickles/5_CMF/CMF_shap.pkl','rb'))
    CMF_rmse  = pickle.load(open('Pickles/5_CMF/CMF_rmse.pkl','rb'))
    #CMF_explainer = shap.TreeExplainer(CMF_model)
    CMF_fs = pd.read_csv('Pickles/5_CMF/CMF_fs.csv')

    return CMF_model, CMF_scaler, CMF_rmse, CMF_explainer, CMF_fs

CMF_model, CMF_scaler, CMF_rmse, CMF_explainer, CMF_fs = read_objects()

# Setting up the page for position CMF
if position == 'Central Midfielder':

    # Setting up the sidebar
    with st.sidebar:
        with st.expander('Expand to fill in player performance'):
            col1, col2 = st.columns(2)
            feature_0 = col1.number_input((CMF_fs.iloc[0].values[0]), key=0, min_value=0, step=10)
            feature_2 = col1.number_input((CMF_fs.iloc[2].values[0]), key=2, min_value=0, step=10)
            feature_4 = col1.number_input((CMF_fs.iloc[4].values[0]), key=4, min_value=0, step=10)
            feature_6 = col1.number_input((CMF_fs.iloc[6].values[0]), key=6, min_value=0, step=200)
            feature_8 = col1.number_input((CMF_fs.iloc[8].values[0]), key=8, min_value=0, step=5)
            
            feature_1 = col2.number_input((CMF_fs.iloc[1].values[0]), key=1, min_value=0, step=10)
            feature_3 = col2.number_input((CMF_fs.iloc[3].values[0]), key=3, min_value=0, step=10)
            feature_5 = col2.number_input((CMF_fs.iloc[5].values[0]), key=5, min_value=0, step=2)
            feature_7 = col2.number_input((CMF_fs.iloc[7].values[0]), key=7, min_value=0, step=10)
            feature_9 = col2.number_input((CMF_fs.iloc[9].values[0]), key=9, min_value=0, step=5)
            
            # Adding a button that triggers prediction of the rating
            predict_button = st.button('Predict Player Rating')

    # Setting up the page
    col1, col2 = st.columns([2, 3])

    # Adding user input from the sidebar to the page
    if player == 'Player': col1.header('Player')
    else: col1.header(player)
    if position == 'Position': col1.info('Position:  ')
    else: col1.info('Position:  ' + position)
    if team == 'Team': col1.info('Team:  ')
    else: col1.info('Team:  ' + team)
    if foot == 'Foot': col1.info('Foot:  ')
    else: col1.info('Foot:  ' + foot)
    if age == 'Age': col1.info('Age:  ')
    else: col1.info('Age:  ' + age)
    if height == 'Height': col1.info('Height:  ')
    else: col1.info('Height:  ' + height)
    if weight == 'Weight': col1.info('Weight:  ')
    else: col1.info('Weight:  ' + weight)

    # Setting up the radar graph
    col2.write('')
    col2.write('')
    categories =    [CMF_fs.iloc[0],
                    CMF_fs.iloc[1],
                    CMF_fs.iloc[2],
                    CMF_fs.iloc[3],
                    CMF_fs.iloc[4],
                    CMF_fs.iloc[5],
                    CMF_fs.iloc[6],
                    CMF_fs.iloc[7],
                    CMF_fs.iloc[8],
                    CMF_fs.iloc[9]]
    fig.add_trace(go.Scatterpolar(r=    [(feature_0 - 0) / (BL[CMF_fs.iloc[0].values[0]].max() - 0) * 100,
                                        (feature_1 - 0) / (BL[CMF_fs.iloc[1].values[0]].max() - 0) * 100,
                                        (feature_2 - 0) / (BL[CMF_fs.iloc[2].values[0]].max() - 0) * 100,
                                        (feature_3 - 0) / (BL[CMF_fs.iloc[3].values[0]].max() - 0) * 100,
                                        (feature_4 - 0) / (BL[CMF_fs.iloc[4].values[0]].max() - 0) * 100,
                                        (feature_5 - 0) / (BL[CMF_fs.iloc[5].values[0]].max() - 0) * 100,
                                        (feature_6 - 0) / (BL[CMF_fs.iloc[6].values[0]].max() - 0) * 100,
                                        (feature_7 - 0) / (BL[CMF_fs.iloc[7].values[0]].max() - 0) * 100,
                                        (feature_8 - 0) / (BL[CMF_fs.iloc[8].values[0]].max() - 0) * 100,
                                        (feature_9 - 0) / (BL[CMF_fs.iloc[9].values[0]].max() - 0) * 100
                                        ],theta=categories, fill='toself', name=player))

    # Plotting the graph
    col2.plotly_chart(fig, use_container_width=True)

    # Setting up the rest of the page
    col1, col2, col3, col4, col5, col6= st.columns([2, 0.8, 1, 0.1, 1, 0.1])

    # Adding user input from the sidebar to the page
    if nationality == 'Nationality': col1.info('Nationality:  ')
    else: col1.info('Nationality:  ' + nationality)
    col2.write('')

    # Creating an if statement that triggers the prediction when pressing the predict_button
    if predict_button:

        # Creating a dataframe with feature names and user input from the sidebar
        user_input = pd.DataFrame({ CMF_fs.iloc[0].values[0]:feature_0,
                                    CMF_fs.iloc[1].values[0]:feature_1, 
                                    CMF_fs.iloc[2].values[0]:feature_2, 
                                    CMF_fs.iloc[3].values[0]:feature_3, 
                                    CMF_fs.iloc[4].values[0]:feature_4, 
                                    CMF_fs.iloc[5].values[0]:feature_5,
                                    CMF_fs.iloc[6].values[0]:feature_6,                                         
                                    CMF_fs.iloc[7].values[0]:feature_7, 
                                    CMF_fs.iloc[8].values[0]:feature_8, 
                                    CMF_fs.iloc[9].values[0]:feature_9, 
                                    }, index=[0]) 
        # Scaling the user input
        user_input_scaled = pd.DataFrame(CMF_scaler.transform(user_input), columns = user_input.columns, index=[0])  

        # Predicting the rating based on the user input
        predicted_rating = CMF_model.predict(user_input_scaled)
    
        # Displaying the rating and RMSE
        col3.metric(label="Player Rating", value=int(predicted_rating))
        col4.write('')
        col5.metric(label="Mean Error", value=np.round(CMF_rmse, decimals = 2))
        col6.write('')

        # Displaying the SHAP values
        #with st.expander("Player Rating Explained"):
        #    shap_value = CMF_explainer.shap_values(user_input_scaled)
        #    st_shap(shap.force_plot(CMF_explainer.expected_value, shap_value, user_input_scaled), height=150, width=700)
    
    # Default display when the button has not been pressed
    else:
        col3.metric(label="Player Rating", value='00')
        col4.write('')
        col5.metric(label="Mean Error", value='00.0')
        col6.write('')
        with st.expander("Player Rating Explained"):
            st.write('')
















# Position AMF: Attacking Midfielder

# Loading models, scalers, explainers, etc., only once
@st.experimental_singleton
def read_objects():
    # Model, scaler, explainer and features selected for each position
    AMF_model = pickle.load(open('Pickles/6_AMF/AMF_model.pkl','rb'))
    AMF_scaler = pickle.load(open('Pickles/6_AMF/AMF_scaler.pkl','rb'))
    #AMF_shap_values = pickle.load(open('Pickles/6_AMF/AMF_shap.pkl','rb'))
    AMF_rmse  = pickle.load(open('Pickles/6_AMF/AMF_rmse.pkl','rb'))
    #AMF_explainer = shap.TreeExplainer(AMF_model)
    AMF_fs = pd.read_csv('Pickles/6_AMF/AMF_fs.csv')

    return AMF_model, AMF_scaler, AMF_rmse, AMF_explainer, AMF_fs

AMF_model, AMF_scaler, AMF_rmse, AMF_explainer, AMF_fs = read_objects()

# Setting up the page for position AMF
if position == 'Attacking Midfielder':

    # Setting up the sidebar
    with st.sidebar:
        with st.expander('Expand to fill in player performance'):
            col1, col2 = st.columns(2)
            feature_0 = col1.number_input((AMF_fs.iloc[0].values[0]), key=0, min_value=0, step=200)
            feature_2 = col1.number_input((AMF_fs.iloc[2].values[0]), key=2, min_value=0, step=1)
            feature_4 = col1.number_input((AMF_fs.iloc[4].values[0]), key=4, min_value=0, step=10)
            feature_6 = col1.number_input((AMF_fs.iloc[6].values[0]), key=6, min_value=0, step=1)
            feature_8 = col1.number_input((AMF_fs.iloc[8].values[0]), key=8, min_value=0, step=5)
            
            feature_1 = col2.number_input((AMF_fs.iloc[1].values[0]), key=1, min_value=0, step=5)
            feature_3 = col2.number_input((AMF_fs.iloc[3].values[0]), key=3, min_value=0, step=5)
            feature_5 = col2.number_input((AMF_fs.iloc[5].values[0]), key=5, min_value=0, step=5)
            feature_7 = col2.number_input((AMF_fs.iloc[7].values[0]), key=7, min_value=0, step=10)
            feature_9 = col2.number_input((AMF_fs.iloc[9].values[0]), key=9, min_value=0, step=1)
            
            # Adding a button that triggers prediction of the rating
            predict_button = st.button('Predict Player Rating')

    # Setting up the page
    col1, col2 = st.columns([2, 3])

    # Adding user input from the sidebar to the page
    if player == 'Player': col1.header('Player')
    else: col1.header(player)
    if position == 'Position': col1.info('Position:  ')
    else: col1.info('Position:  ' + position)
    if team == 'Team': col1.info('Team:  ')
    else: col1.info('Team:  ' + team)
    if foot == 'Foot': col1.info('Foot:  ')
    else: col1.info('Foot:  ' + foot)
    if age == 'Age': col1.info('Age:  ')
    else: col1.info('Age:  ' + age)
    if height == 'Height': col1.info('Height:  ')
    else: col1.info('Height:  ' + height)
    if weight == 'Weight': col1.info('Weight:  ')
    else: col1.info('Weight:  ' + weight)

    # Setting up the radar graph
    col2.write('')
    col2.write('')
    categories =    [AMF_fs.iloc[0],
                    AMF_fs.iloc[1],
                    AMF_fs.iloc[2],
                    AMF_fs.iloc[3],
                    AMF_fs.iloc[4],
                    AMF_fs.iloc[5],
                    AMF_fs.iloc[6],
                    AMF_fs.iloc[7],
                    AMF_fs.iloc[8],
                    AMF_fs.iloc[9]]
    fig.add_trace(go.Scatterpolar(r=    [(feature_0 - 0) / (BL[AMF_fs.iloc[0].values[0]].max() - 0) * 100,
                                        (feature_1 - 0) / (BL[AMF_fs.iloc[1].values[0]].max() - 0) * 100,
                                        (feature_2 - 0) / (BL[AMF_fs.iloc[2].values[0]].max() - 0) * 100,
                                        (feature_3 - 0) / (BL[AMF_fs.iloc[3].values[0]].max() - 0) * 100,
                                        (feature_4 - 0) / (BL[AMF_fs.iloc[4].values[0]].max() - 0) * 100,
                                        (feature_5 - 0) / (BL[AMF_fs.iloc[5].values[0]].max() - 0) * 100,
                                        (feature_6 - 0) / (BL[AMF_fs.iloc[6].values[0]].max() - 0) * 100,
                                        (feature_7 - 0) / (BL[AMF_fs.iloc[7].values[0]].max() - 0) * 100,
                                        (feature_8 - 0) / (BL[AMF_fs.iloc[8].values[0]].max() - 0) * 100,
                                        (feature_9 - 0) / (BL[AMF_fs.iloc[9].values[0]].max() - 0) * 100
                                        ],theta=categories, fill='toself', name=player))

    # Plotting the graph
    col2.plotly_chart(fig, use_container_width=True)

    # Setting up the rest of the page
    col1, col2, col3, col4, col5, col6= st.columns([2, 0.8, 1, 0.1, 1, 0.1])

    # Adding user input from the sidebar to the page
    if nationality == 'Nationality': col1.info('Nationality:  ')
    else: col1.info('Nationality:  ' + nationality)
    col2.write('')

    # Creating an if statement that triggers the prediction when pressing the predict_button
    if predict_button:

        # Creating a dataframe with feature names and user input from the sidebar
        user_input = pd.DataFrame({ AMF_fs.iloc[0].values[0]:feature_0,
                                    AMF_fs.iloc[1].values[0]:feature_1, 
                                    AMF_fs.iloc[2].values[0]:feature_2, 
                                    AMF_fs.iloc[3].values[0]:feature_3, 
                                    AMF_fs.iloc[4].values[0]:feature_4, 
                                    AMF_fs.iloc[5].values[0]:feature_5,
                                    AMF_fs.iloc[6].values[0]:feature_6,                                         
                                    AMF_fs.iloc[7].values[0]:feature_7, 
                                    AMF_fs.iloc[8].values[0]:feature_8, 
                                    AMF_fs.iloc[9].values[0]:feature_9, 
                                    }, index=[0]) 
        # Scaling the user input
        user_input_scaled = pd.DataFrame(AMF_scaler.transform(user_input), columns = user_input.columns, index=[0])  

        # Predicting the rating based on the user input
        predicted_rating = AMF_model.predict(user_input_scaled)
    
        # Displaying the rating and RMSE
        col3.metric(label="Player Rating", value=int(predicted_rating))
        col4.write('')
        col5.metric(label="Mean Error", value=np.round(AMF_rmse, decimals = 2))
        col6.write('')

        # Displaying the SHAP values
        #with st.expander("Player Rating Explained"):
        #    shap_value = AMF_explainer.shap_values(user_input_scaled)
        #    st_shap(shap.force_plot(AMF_explainer.expected_value, shap_value, user_input_scaled), height=150, width=700)
    
    # Default display when the button has not been pressed
    else:
        col3.metric(label="Player Rating", value='00')
        col4.write('')
        col5.metric(label="Mean Error", value='00.0')
        col6.write('')
        with st.expander("Player Rating Explained"):
            st.write('')















# Position WMF: Winger Midfielder

# Loading models, scalers, explainers, etc., only once
@st.experimental_singleton
def read_objects():
    # Model, scaler, explainer and features selected for each position
    WMF_model = pickle.load(open('Pickles/7_WMF/WMF_model.pkl','rb'))
    WMF_scaler = pickle.load(open('Pickles/7_WMF/WMF_scaler.pkl','rb'))
    #WMF_shap_values = pickle.load(open('Pickles/7_WMF/WMF_shap.pkl','rb'))
    WMF_rmse  = pickle.load(open('Pickles/7_WMF/WMF_rmse.pkl','rb'))
    #WMF_explainer = shap.TreeExplainer(WMF_model)
    WMF_fs = pd.read_csv('Pickles/7_WMF/WMF_fs.csv')

    return WMF_model, WMF_scaler, WMF_rmse, WMF_explainer, WMF_fs

WMF_model, WMF_scaler, WMF_rmse, WMF_explainer, WMF_fs = read_objects()

# Setting up the page for position WMF
if position == 'Winger Midfielder':

    # Setting up the sidebar
    with st.sidebar:
        with st.expander('Expand to fill in player performance'):
            col1, col2 = st.columns(2)
            feature_0 = col1.number_input((WMF_fs.iloc[0].values[0]), key=0, min_value=0, step=5)
            feature_2 = col1.number_input((WMF_fs.iloc[2].values[0]), key=2, min_value=0, step=10)
            feature_4 = col1.number_input((WMF_fs.iloc[4].values[0]), key=4, min_value=0, step=1)
            feature_6 = col1.number_input((WMF_fs.iloc[6].values[0]), key=6, min_value=0, step=200)
            feature_8 = col1.number_input((WMF_fs.iloc[8].values[0]), key=8, min_value=0, step=1)
            
            feature_1 = col2.number_input((WMF_fs.iloc[1].values[0]), key=1, min_value=0, step=1)
            feature_3 = col2.number_input((WMF_fs.iloc[3].values[0]), key=3, min_value=0, step=10)
            feature_5 = col2.number_input((WMF_fs.iloc[5].values[0]), key=5, min_value=0, step=10)
            feature_7 = col2.number_input((WMF_fs.iloc[7].values[0]), key=7, min_value=0, step=1)
            feature_9 = col2.number_input((WMF_fs.iloc[9].values[0]), key=9, min_value=0, step=5)
            
            # Adding a button that triggers prediction of the rating
            predict_button = st.button('Predict Player Rating')

    # Setting up the page
    col1, col2 = st.columns([2, 3])

    # Adding user input from the sidebar to the page
    if player == 'Player': col1.header('Player')
    else: col1.header(player)
    if position == 'Position': col1.info('Position:  ')
    else: col1.info('Position:  ' + position)
    if team == 'Team': col1.info('Team:  ')
    else: col1.info('Team:  ' + team)
    if foot == 'Foot': col1.info('Foot:  ')
    else: col1.info('Foot:  ' + foot)
    if age == 'Age': col1.info('Age:  ')
    else: col1.info('Age:  ' + age)
    if height == 'Height': col1.info('Height:  ')
    else: col1.info('Height:  ' + height)
    if weight == 'Weight': col1.info('Weight:  ')
    else: col1.info('Weight:  ' + weight)

    # Setting up the radar graph
    col2.write('')
    col2.write('')
    categories =    [WMF_fs.iloc[0],
                    WMF_fs.iloc[1],
                    WMF_fs.iloc[2],
                    WMF_fs.iloc[3],
                    WMF_fs.iloc[4],
                    WMF_fs.iloc[5],
                    WMF_fs.iloc[6],
                    WMF_fs.iloc[7],
                    WMF_fs.iloc[8],
                    WMF_fs.iloc[9]]
    fig.add_trace(go.Scatterpolar(r=    [(feature_0 - 0) / (BL[WMF_fs.iloc[0].values[0]].max() - 0) * 100,
                                        (feature_1 - 0) / (BL[WMF_fs.iloc[1].values[0]].max() - 0) * 100,
                                        (feature_2 - 0) / (BL[WMF_fs.iloc[2].values[0]].max() - 0) * 100,
                                        (feature_3 - 0) / (BL[WMF_fs.iloc[3].values[0]].max() - 0) * 100,
                                        (feature_4 - 0) / (BL[WMF_fs.iloc[4].values[0]].max() - 0) * 100,
                                        (feature_5 - 0) / (BL[WMF_fs.iloc[5].values[0]].max() - 0) * 100,
                                        (feature_6 - 0) / (BL[WMF_fs.iloc[6].values[0]].max() - 0) * 100,
                                        (feature_7 - 0) / (BL[WMF_fs.iloc[7].values[0]].max() - 0) * 100,
                                        (feature_8 - 0) / (BL[WMF_fs.iloc[8].values[0]].max() - 0) * 100,
                                        (feature_9 - 0) / (BL[WMF_fs.iloc[9].values[0]].max() - 0) * 100
                                        ],theta=categories, fill='toself', name=player))

    # Plotting the graph
    col2.plotly_chart(fig, use_container_width=True)

    # Setting up the rest of the page
    col1, col2, col3, col4, col5, col6= st.columns([2, 0.8, 1, 0.1, 1, 0.1])

    # Adding user input from the sidebar to the page
    if nationality == 'Nationality': col1.info('Nationality:  ')
    else: col1.info('Nationality:  ' + nationality)
    col2.write('')

    # Creating an if statement that triggers the prediction when pressing the predict_button
    if predict_button:

        # Creating a dataframe with feature names and user input from the sidebar
        user_input = pd.DataFrame({ WMF_fs.iloc[0].values[0]:feature_0,
                                    WMF_fs.iloc[1].values[0]:feature_1, 
                                    WMF_fs.iloc[2].values[0]:feature_2, 
                                    WMF_fs.iloc[3].values[0]:feature_3, 
                                    WMF_fs.iloc[4].values[0]:feature_4, 
                                    WMF_fs.iloc[5].values[0]:feature_5,
                                    WMF_fs.iloc[6].values[0]:feature_6,                                         
                                    WMF_fs.iloc[7].values[0]:feature_7, 
                                    WMF_fs.iloc[8].values[0]:feature_8, 
                                    WMF_fs.iloc[9].values[0]:feature_9, 
                                    }, index=[0]) 
        # Scaling the user input
        user_input_scaled = pd.DataFrame(WMF_scaler.transform(user_input), columns = user_input.columns, index=[0])  

        # Predicting the rating based on the user input
        predicted_rating = WMF_model.predict(user_input_scaled)
    
        # Displaying the rating and RMSE
        col3.metric(label="Player Rating", value=int(predicted_rating))
        col4.write('')
        col5.metric(label="Mean Error", value=np.round(WMF_rmse, decimals = 2))
        col6.write('')

        # Displaying the SHAP values
        #with st.expander("Player Rating Explained"):
        #    shap_value = WMF_explainer.shap_values(user_input_scaled)
        #    st_shap(shap.force_plot(WMF_explainer.expected_value, shap_value, user_input_scaled), height=150, width=700)
    
    # Default display when the button has not been pressed
    else:
        col3.metric(label="Player Rating", value='00')
        col4.write('')
        col5.metric(label="Mean Error", value='00.0')
        col6.write('')
        with st.expander("Player Rating Explained"):
            st.write('')















# Position FW: Forwarder

# Loading models, scalers, explainers, etc., only once
@st.experimental_singleton
def read_objects():
    # Model, scaler, explainer and features selected for each position
    FW_model = pickle.load(open('Pickles/8_FW/FW_model.pkl','rb'))
    FW_scaler = pickle.load(open('Pickles/8_FW/FW_scaler.pkl','rb'))
    #FW_shap_values = pickle.load(open('Pickles/8_FW/FW_shap.pkl','rb'))
    FW_rmse  = pickle.load(open('Pickles/8_FW/FW_rmse.pkl','rb'))
    #FW_explainer = shap.TreeExplainer(FW_model)
    FW_fs = pd.read_csv('Pickles/8_FW/FW_fs.csv')

    return FW_model, FW_scaler, FW_rmse, FW_explainer, FW_fs

FW_model, FW_scaler, FW_rmse, FW_explainer, FW_fs = read_objects()

# Setting up the page for position FW
if position == 'Forwarder':

    # Setting up the sidebar
    with st.sidebar:
        with st.expander('Expand to fill in player performance'):
            col1, col2 = st.columns(2)
            feature_0 = col1.number_input((FW_fs.iloc[0].values[0]), key=0, min_value=0, step=5)
            feature_2 = col1.number_input((FW_fs.iloc[2].values[0]), key=2, min_value=0, step=5)
            feature_4 = col1.number_input((FW_fs.iloc[4].values[0]), key=4, min_value=0, step=10)
            feature_6 = col1.number_input((FW_fs.iloc[6].values[0]), key=6, min_value=0, step=10)
            feature_8 = col1.number_input((FW_fs.iloc[8].values[0]), key=8, min_value=0, step=1)
            
            feature_1 = col2.number_input((FW_fs.iloc[1].values[0]), key=1, min_value=0, step=5)
            feature_3 = col2.number_input((FW_fs.iloc[3].values[0]), key=3, min_value=0, step=2)
            feature_5 = col2.number_input((FW_fs.iloc[5].values[0]), key=5, min_value=0, step=1)
            feature_7 = col2.number_input((FW_fs.iloc[7].values[0]), key=7, min_value=0, step=200)
            feature_9 = col2.number_input((FW_fs.iloc[9].values[0]), key=9, min_value=0, step=2)
            
            # Adding a button that triggers prediction of the rating
            predict_button = st.button('Predict Player Rating')

    # Setting up the page
    col1, col2 = st.columns([2, 3])

    # Adding user input from the sidebar to the page
    if player == 'Player': col1.header('Player')
    else: col1.header(player)
    if position == 'Position': col1.info('Position:  ')
    else: col1.info('Position:  ' + position)
    if team == 'Team': col1.info('Team:  ')
    else: col1.info('Team:  ' + team)
    if foot == 'Foot': col1.info('Foot:  ')
    else: col1.info('Foot:  ' + foot)
    if age == 'Age': col1.info('Age:  ')
    else: col1.info('Age:  ' + age)
    if height == 'Height': col1.info('Height:  ')
    else: col1.info('Height:  ' + height)
    if weight == 'Weight': col1.info('Weight:  ')
    else: col1.info('Weight:  ' + weight)

    # Setting up the radar graph
    col2.write('')
    col2.write('')
    categories =    [FW_fs.iloc[0],
                    FW_fs.iloc[1],
                    FW_fs.iloc[2],
                    FW_fs.iloc[3],
                    FW_fs.iloc[4],
                    FW_fs.iloc[5],
                    FW_fs.iloc[6],
                    FW_fs.iloc[7],
                    FW_fs.iloc[8],
                    FW_fs.iloc[9]]
    fig.add_trace(go.Scatterpolar(r=    [(feature_0 - 0) / (BL[FW_fs.iloc[0].values[0]].max() - 0) * 100,
                                        (feature_1 - 0) / (BL[FW_fs.iloc[1].values[0]].max() - 0) * 100,
                                        (feature_2 - 0) / (BL[FW_fs.iloc[2].values[0]].max() - 0) * 100,
                                        (feature_3 - 0) / (BL[FW_fs.iloc[3].values[0]].max() - 0) * 100,
                                        (feature_4 - 0) / (BL[FW_fs.iloc[4].values[0]].max() - 0) * 100,
                                        (feature_5 - 0) / (BL[FW_fs.iloc[5].values[0]].max() - 0) * 100,
                                        (feature_6 - 0) / (BL[FW_fs.iloc[6].values[0]].max() - 0) * 100,
                                        (feature_7 - 0) / (BL[FW_fs.iloc[7].values[0]].max() - 0) * 100,
                                        (feature_8 - 0) / (BL[FW_fs.iloc[8].values[0]].max() - 0) * 100,
                                        (feature_9 - 0) / (BL[FW_fs.iloc[9].values[0]].max() - 0) * 100
                                        ],theta=categories, fill='toself', name=player))

    # Plotting the graph
    col2.plotly_chart(fig, use_container_width=True)

    # Setting up the rest of the page
    col1, col2, col3, col4, col5, col6= st.columns([2, 0.8, 1, 0.1, 1, 0.1])

    # Adding user input from the sidebar to the page
    if nationality == 'Nationality': col1.info('Nationality:  ')
    else: col1.info('Nationality:  ' + nationality)
    col2.write('')

    # Creating an if statement that triggers the prediction when pressing the predict_button
    if predict_button:

        # Creating a dataframe with feature names and user input from the sidebar
        user_input = pd.DataFrame({ FW_fs.iloc[0].values[0]:feature_0,
                                    FW_fs.iloc[1].values[0]:feature_1, 
                                    FW_fs.iloc[2].values[0]:feature_2, 
                                    FW_fs.iloc[3].values[0]:feature_3, 
                                    FW_fs.iloc[4].values[0]:feature_4, 
                                    FW_fs.iloc[5].values[0]:feature_5,
                                    FW_fs.iloc[6].values[0]:feature_6,                                         
                                    FW_fs.iloc[7].values[0]:feature_7, 
                                    FW_fs.iloc[8].values[0]:feature_8, 
                                    FW_fs.iloc[9].values[0]:feature_9, 
                                    }, index=[0]) 
        # Scaling the user input
        user_input_scaled = pd.DataFrame(FW_scaler.transform(user_input), columns = user_input.columns, index=[0])  

        # Predicting the rating based on the user input
        predicted_rating = FW_model.predict(user_input_scaled)
    
        # Displaying the rating and RMSE
        col3.metric(label="Player Rating", value=int(predicted_rating))
        col4.write('')
        col5.metric(label="Mean Error", value=np.round(FW_rmse, decimals = 2))
        col6.write('')

        # Displaying the SHAP values
        #with st.expander("Player Rating Explained"):
        #    shap_value = FW_explainer.shap_values(user_input_scaled)
        #    st_shap(shap.force_plot(FW_explainer.expected_value, shap_value, user_input_scaled), height=150, width=700)
    
    # Default display when the button has not been pressed
    else:
        col3.metric(label="Player Rating", value='00')
        col4.write('')
        col5.metric(label="Mean Error", value='00.0')
        col6.write('')
        with st.expander("Player Rating Explained"):
            st.write('')
