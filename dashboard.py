import streamlit as st
from streamlit_option_menu import option_menu
from numerize import numerize
import pandas as pd
import datetime as dt
import plotly.express as px
import numpy as np

st.set_page_config(layout="wide",
                   page_title = 'Dashboard Freshworks',
                   page_icon = ':white_check_mark:')

mesi =['Gennaio', 'Febbraio', 'Marzo', 'Aprile', 
       'Maggio', 'Giugno', 'Luglio', 'Agosto', 
       'Settembre', 'Ottobre', 'Novembre', 'Dicembre']

mon = {'Gennaio':1,
       'Febbraio':2,
       'Marzo':3,
       'Aprile':4,
       'Maggio':5,
       'Giugno':6,
       'Luglio':7,
       'Agosto':8,
       'Settembre':9,
       'Ottobre':10,
       'Novembre':11,
       'Dicembre':12}

quart = ['Q1','Q2','Q3','Q4']

df = pd.read_csv('dati.csv')

df['INVOICE DATE'] = pd.to_datetime(df['INVOICE DATE'])
df['PAID ON DATE'] = pd.to_datetime(df['PAID ON DATE'])

df.index = df['PAID ON DATE']

df['Year'] = df['PAID ON DATE'].dt.year
df['Month'] = df['PAID ON DATE'].dt.month
df['Day of Year'] = df['PAID ON DATE'].dt.dayofyear
df['Quarter'] = df['PAID ON DATE'].dt.quarter
df['Days from Today'] = (dt.datetime.today()-df['PAID ON DATE']).dt.days

selected = option_menu("Seleziona Visualizzazione", ["Dashboard Finanziaria", 'Controllo Pagamenti'], 
        icons=['currency-dollar', 'gear'], default_index=0,
        menu_icon="binoculars", orientation="horizontal")

if selected == 'Dashboard Finanziaria':
    st.markdown("""# Kahuna Performance Finanziaria :white_check_mark:""")
    
    with st.sidebar:
        st.title('Selezionare Periodo')
        q_or_no = st.checkbox('Quarterly?')
        
        pp = quart if q_or_no else mesi
        s = 'Trimestre' if q_or_no else 'Mese'
        
        t_var = st.selectbox(s, pp, index = 0)
        y_var = st.selectbox('Anno',[2017,2018,2019,2020,2021,2022],index = 5)
        
        month = mon[t_var] if not q_or_no else ''
        quarter = quart.index(t_var) + 1 if q_or_no else ''
        year = y_var
            
    st.markdown(f'## Dashboard finanziaria per il periodo: {quart[quarter-1] if q_or_no else mesi[month-1]} del {year}')
    
    one, two = st.columns(2)
    two.markdown('## Widget Riassuntivi')
    
    letter = 'Q' if q_or_no else 'M'
    come = 'Trimestrale' if q_or_no else 'Mensile'
    period = 4 if q_or_no else 12
    come1 = 'Trimestre' if q_or_no else 'Mese'
    
    mon_values = df[['AMOUNT (USD)','COMMISSION (USD)']].resample(letter).sum()
    mon_values['Month'] = mon_values.index.month
    mon_values['Year'] = mon_values.index.year
    
    a = mon_values.pivot(index = 'Month', columns = 'Year', values = 'COMMISSION (USD)')
    a['Mese'] = quart if q_or_no else mon.keys()
    a['Improvement'] = (a[year]/a[year-1]) -1
    a['Holder'] = np.where(a['Improvement']>0,1,0)
    a['Holder 2'] = [str(round(i*100,2))+'%' for i in a['Improvement'].values]
    
    pct_change = mon_values.iloc[-1]/mon_values.iloc[-2] - 1
    ann = mon_values.iloc[-1]/mon_values.iloc[-period-1] -1
    
    percent_anno = (df[df['Year'] == year][['AMOUNT (USD)', 'COMMISSION (USD)']]\
                            .sum(numeric_only = True)/df[df['Year'] == year-1]\
                            [['AMOUNT (USD)', 'COMMISSION (USD)']].sum(numeric_only = True)) - 1
    
    guad_anno = df[df['Year'] == year][['AMOUNT (USD)', 'COMMISSION (USD)']].sum(numeric_only = True)
    
    if q_or_no:
        guad_mese = df[(df['Year'] == year) & 
                       (df['Quarter'] == quarter)][['AMOUNT (USD)', 
                                                    'COMMISSION (USD)']].sum(numeric_only = True)
    else:    
        guad_mese = df[(df['Year'] == year) & 
                       (df['Month'] == month)][['AMOUNT (USD)', 
                                                'COMMISSION (USD)']].sum(numeric_only = True)
    
    plt1 = px.line(mon_values,
                   x = mon_values.index, 
                   y = ['COMMISSION (USD)','AMOUNT (USD)'],
                   title = f'Andamento {come} Venduto e Commissioni',
                   labels = {'value':'Importo','PAID ON DATE':'Data'})
    
    plt1.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
    plt1.update_layout(
        xaxis = dict(
            showline = False,
            showgrid = False,
        ),
        yaxis = dict(
            showline = True,
            showgrid = False,
        )
    )
    
    plt2 = px.line(a, 
                   x = 'Mese', 
                   y = [year-1, year], 
                   title = f'Andamento {come} {year} vs. {year-1}',
                   labels = {'value':'Fatturato'})
    
    plt2.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                        'paper_bgcolor': 'rgba(0, 0, 0, 0)',})
    
    plt2.update_layout(
        xaxis = dict(
            showline = False,
            showgrid = False,
        ),
        yaxis = dict(
            showline = True,
            showgrid = False,
        )
    )
    
    plt2_2 = px.bar(a, 
                    x = 'Mese', 
                    y = [year-1, year], 
                    barmode='group',
                    title = f'Andamento {come} {year} vs. {year-1}',
                    labels = {'value':'Fatturato'})
    
    plt2_2.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
    plt2_2.update_layout(
        xaxis = dict(
            showline = False,
            showgrid = False,
        ),
        yaxis = dict(
            showline = True,
            showgrid = False,
        )
    )
    
    improvement_mensili = px.bar(a,
                                 x = 'Mese', 
                                 y = 'Improvement', 
                                 color = 'Improvement',
                                 title = f'Miglioramento {come} Rispetto al {year-1}',
                                 text = 'Holder 2',
                                 color_continuous_scale = 'blues')
    
    improvement_mensili.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    improvement_mensili.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                                       'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    improvement_mensili.update_layout(
        xaxis = dict(
            showline = False,
            showgrid = False,
        ),
        yaxis = dict(
            showline = True,
            showgrid = False,
        )
    )
    
    colA, colB, colC = st.columns([2,1,1])
    
    day_today = df[df['Year'] == year]['Day of Year'].max()
    hello = df[(df['Year'] == year-1) & 
               (df['Day of Year']<= day_today)][['AMOUNT (USD)', 
                                                 'COMMISSION (USD)']].sum(numeric_only = True)
    nope = df[df['Year'] == year][['AMOUNT (USD)', 'COMMISSION (USD)']].sum(numeric_only = True)/hello -1
    
    thing = quart[quarter - 1] if q_or_no else mesi[month-1]
    
    with colA:
        st.plotly_chart(plt1,use_container_width=True)
    with colB:
        st.metric(f'Venduto {year}',
                 f'${numerize.numerize(guad_anno["AMOUNT (USD)"],2)}')
        st.metric(f'Fatturato {thing} {year} vs {come1} Precedente',
                  f'${numerize.numerize(guad_mese["AMOUNT (USD)"],2)}',
                  delta = str(round(pct_change['AMOUNT (USD)']*100,2)) + '%')
        st.metric(f'Percentuale del Fatturato {year-1}',
                  f'{round(percent_anno["AMOUNT (USD)"]*100,2)}%')
        st.metric(f'Variazione Fatturato Stesso Periodo {year-1}',
                  f'{round(nope["AMOUNT (USD)"]*100,2)}%')
        
    with colC:
        st.metric(f'Commissioni Generate nel {year}',
                 f"${numerize.numerize(guad_anno['COMMISSION (USD)'],2)}")
        st.metric(f'Commissioni {thing} {year} vs {come1} Precedente',
                  f"${numerize.numerize(guad_mese['COMMISSION (USD)'],2)}",
                  delta = str(round(pct_change['COMMISSION (USD)']*100,2)) + '%')
        st.metric(f'Percentuale delle Commissioni Generate nel {year-1}',
                  f'{round(percent_anno["COMMISSION (USD)"]*100,2)}%')
        st.metric(f'Variazione Commissioni Stesso Periodo {year-1}',
                  f'{round(nope["COMMISSION (USD)"]*100,2)}%')
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not q_or_no:
            st.plotly_chart(plt2,use_container_width=True)
        else:
            st.plotly_chart(plt2_2,use_container_width=True)
    
    with col2:
        st.plotly_chart(improvement_mensili,use_container_width=True)
else:
    st.markdown("""# Controllo Pagamenti Fresh :flag-in:""")  
    
    new_pmts = pd.read_csv('pmts.csv')
    paid_codes = new_pmts['INVOICE CODE'].tolist()

    not_paid = df[~df['INVOICE #'].isin(paid_codes)]
    
    CURRENT_MONTH = 3
    CURRENT_YEAR = 2022
    
    not_paid = not_paid[(not_paid['Month'] != CURRENT_MONTH) & 
                        (not_paid['Year'] != CURRENT_YEAR) &
                        (not_paid['AMOUNT (USD)']>0)]
    
    c1, c2, c3, c4, c5 = st.columns([1,1,0.75,0.75,0.75])
    
    showing = not_paid[['INVOICE #', 'ACCOUNT', 'PAID ON DATE', 
                        'INVOICE DATE','AMOUNT (USD)',
                        'COMMISSION (USD)', 'Days from Today']].copy()
    
    showing.index = showing['INVOICE #']
   
    rge = c1.slider('Giorni da Oggi',
                    showing['Days from Today'].min(),
                    showing['Days from Today'].max(),
                    (showing['Days from Today'].min(),
                    showing['Days from Today'].max()))
    
    rge1 = c2.slider('Importo',
                    showing['COMMISSION (USD)'].min(),
                    showing['COMMISSION (USD)'].max(),
                    (showing['COMMISSION (USD)'].min(),
                    showing['COMMISSION (USD)'].max()))
   
    showing = showing[(showing['Days from Today']>rge[0])&
                      (showing['Days from Today']<rge[1])&
                      (showing['COMMISSION (USD)']>rge1[0])&
                      (showing['COMMISSION (USD)']<rge1[1])]
    
    c3.metric('Totale Commissioni Mancanti',
             f'${numerize.numerize(showing["COMMISSION (USD)"].sum())}')
    
    c4.metric('Numero Fatture Mancanti',
              showing.ACCOUNT.shape[0])
    
    c5.metric('Elapsed Medio',
              round(showing['Days from Today'].mean()))
    
    csv = showing[['ACCOUNT', 'PAID ON DATE', 
                   'INVOICE DATE','AMOUNT (USD)',
                   'COMMISSION (USD)']].to_csv().encode('utf-8')
    
    data = str(dt.datetime.today()).split()[0]
    
    st.download_button(
       "Scarica Selezione",
       csv,
       f"export_{data}.csv",
       "text/csv",
       key='download-csv'
    )
    
    
    st.dataframe(showing[['ACCOUNT', 'PAID ON DATE', 
                        'INVOICE DATE','AMOUNT (USD)',
                        'COMMISSION (USD)']])
    
    
    
    
    
    
# !ngrok authtoken 27HYMfwsk6v8tBLfPzaoKxXMX6O_7HiKZgw6A7qZyxrcvndkr

    
    
    
