# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 20:07:05 2021

@author: Asus
"""



#-------------------------------- Streamlit -------------------------------------






#PACKAGE IMPORT
import pandas as pd
from geopy.geocoders import Nominatim
import streamlit as st
import numpy as np
from deep_translator import GoogleTranslator
from collections import Counter



#LAYOUT
#st.set_page_config(layout="wide")

col1, col2, col3 = st.columns(3)

col1.write(" ")

col2.image('data:image/png;base64,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')

col3.write(" ")



username = "thesustainablesadmin"
password = "streamlit123"


st.title(" ")
st.title('The Sustainables - Analytics')

username_input = st.text_input("Username:", "") 
password_input = st.text_input("Password:", "")

if (username_input != username) or (password_input != password):
    st.error("The credentials are not correct")
else:

    #--------------------------------------------------------DATA PREP---------------------------------------------------------
    
    
    
    #Importing the data
    
    shopify_upload = st.file_uploader("Upload Shopify CSV",
                                   type = 'csv',
                                   accept_multiple_files = False)
    
    if shopify_upload is not None:
        shopify_df = pd.read_csv(shopify_upload)
    else:
        shopify_df = pd.DataFrame(data=None, columns = ["Order ID",
                           "Channel",
                           "Order Date",
                           "City",
                           "Country",
                           "Product Name",
                           "SKU",
                           "Quantity",
                           "Wholesale Price",
                           "Gross Sales",
                           "Discounts",
                           "Net Sales",
                           "Sales",
                           "Returns",
                           "Shipping",
                           "Taxes"])
        
        
        
        
    ankorstore_upload = st.file_uploader("Upload Ankorstore CSV",
                                   type = 'csv',
                                   accept_multiple_files = False)
    
    if ankorstore_upload is not None:
        ankorstore_df = pd.read_csv(ankorstore_upload)
    else:
        ankorstore_df = pd.DataFrame(data=None, columns = ["Order ID",
                           "Channel",
                           "Order Date",
                           "City",
                           "Country",
                           "Product Name",
                           "SKU",
                           "Quantity",
                           "Wholesale Price",
                           "Gross Sales",
                           "Discounts",
                           "Net Sales",
                           "Sales",
                           "Returns",
                           "Shipping",
                           "Taxes"])
    
    
    
    avocadostore_upload = st.file_uploader("Upload Avocado store Excel",
                                   type = 'xlsx',
                                   accept_multiple_files = False)
    
    if avocadostore_upload is not None:
        avocadostore_df = pd.read_excel(avocadostore_upload)
    else:
        avocadostore_df = pd.DataFrame(data=None, columns = ["Order ID",
                           "Channel",
                           "Order Date",
                           "City",
                           "Country",
                           "Product Name",
                           "SKU",
                           "Quantity",
                           "Wholesale Price",
                           "Gross Sales",
                           "Discounts",
                           "Net Sales",
                           "Sales",
                           "Returns",
                           "Shipping",
                           "Taxes"])
        
        
        
    
    shopify_visits = st.file_uploader("Upload Shopify Visits CSV",
                                   type = 'csv',
                                   accept_multiple_files = False)
    
    if shopify_visits is not None:
        visits_df = pd.read_csv(shopify_visits)
#    else:
#        visits_df = pd.DataFrame(data=None, columns = ['day',
#                                                       'ua_browser_version',
#                                                       'ua_browser',
#                                                       'ua_form_factor',
#                                                       'ua_os_version',
#                                                       'ua_os',
#                                                       'page_path',
#                                                       'page_resource_id',
#                                                       'page_type',
#                                                       'page_url',
#                                                       'location_city',
#                                                       'location_region',
#                                                       'location_country',
#                                                       'marketing_event_target',
#                                                       'marketing_event_type',
#                                                       'utm_campaign_content',
#                                                       'utm_campaign_medium',
#                                                       'utm_campaign_name',
#                                                       'utm_campaign_source',
#                                                       'utm_campaign_term',
#                                                       'referrer_host',
#                                                       'referrer_name',
#                                                       'referrer_path',
#                                                       'referrer_source',
#                                                       'referrer_terms',
#                                                       'referrer_url',
#                                                       'total_sessions',
#                                                       'total_carts',
#                                                       'total_checkouts',
#                                                       'total_orders_placed',
#                                                       'total_conversion',
#                                                       'avg_duration',
#                                                       'total_bounce_rate',
#                                                       'total_pageviews',
#                                                       'total_visitors'])
        
        
    
    shopify = shopify_df
    ankorstore = ankorstore_df
    avocado = avocadostore_df
    
    
    if shopify_visits is not None:
        visits_full = visits_df
    else:
        visits_full=None
    
        
    #ankorstore = pd.read_csv("G:/My Drive/Fred/4. Education/Ramon Llull University, ESADE/Academic/Courses/Final project/The Sustainables/Ankorstore Data - Sales.csv")
    #shopify = pd.read_csv("G:/My Drive/Fred/4. Education/Ramon Llull University, ESADE/Academic/Courses/Final project/The Sustainables/Shopify Data - Sales.csv")
    #avocado = pd.read_excel("G:/My Drive/Fred/4. Education/Ramon Llull University, ESADE/Academic/Courses/Final project/The Sustainables/Avocadostore Orders.xlsx",
    #                      index_col=None)
    
    
    #DATA TREATMENT BY STORE
    
    #FUNTIONS
    
    #Formatting all numbers as floats
    def remove_eurosign(df, column):
        new_n=[]
        for number in list(df[column]):
            new_n.append(float(number[1:]))
        df[column]=new_n
        return "Done!"    
    
    #Get countries from avocado store (standardizing countries)
    def get_country(df):
        geolocator = Nominatim(user_agent="FredAF")
    
        cities = df['City']
        countries = []
        for city in cities:
            location = geolocator.geocode(city)
            country = location.address.split(",")[-1].split(" ")[-1]
            countries.append(country)
        
        for i, country in enumerate(countries):
            countries[i] = GoogleTranslator(source='auto',
                                   target='en').translate(country)
        df['Country']=countries
        return "Done!"
    
    #Standardize numbers
    def char_stand(df):
        net_sales=[]
        sales=[]
        shipping=[]
        for i, row in df.iterrows():
            net_sales.append(row['Net Sales'].replace(',','.'))
            sales.append(row['Sales'].replace(',','.'))
            shipping.append(row['Shipping'].replace(',','.'))
        df['Net Sales']=net_sales
        df['Sales']=sales
        df['Shipping']=shipping       
    
    
    #Product Dictionary
    
    products = ['mirage',
                'save the wildlife',
                'save the plants',
                'cloudcastle',
                'granito',
                'an apple a day',
                'save the gardens',
                'save the jungle'
                ]
    
    #Product Tagging
    def product_tagging(df):
        
        sku=[]
    
        for words in list(df['Product Name']):
            ls=[]
            for product in products:
                if product in words.lower():
                    ls.append(product)
            sku.append(ls)        
            
        df['SKU']=sku    
        
        return 'Done!'
    
    
    def product_tagging_avocado(df):
        
        sku=[]
    
        for words in list(df['SKU']):
            ls=[]
            for product in products:
                if product in words.lower():
                    ls.append(product)
            sku.append(ls)        
            
        df['SKU']=sku    
        
        return 'Done!'
    
    
    
    
    
    
    
    #if shopify_upload is not None and ankorstore_upload is not None and avocadostore_upload is not None and shopify_visits is not None:
    
    
        #------------------------------------------------------SHOPIFY------------------------------------------------------
        
        
    if shopify_upload is not None:
        shopify.drop(columns=["Sale ID",
                              "Order",
                              "Transaction type",
                              "Sale type",
                              "POS location",
                              "Billing region",
                              "Shipping region",
                              "Shipping city",
                              "Shipping country",
                              "Product type",
                              "Product vendor",
                              "Variant"],
                     inplace=True
                     )
        
        shopify.rename(columns = {'Date':'Order Date',
                        'Billing country':'Country',          
                        'Billing city':'City',
                        'Product':'Product Name',
                        'Variant SKU':'SKU',
                        'Net quantity':'Quantity',
                        'Gross sales':'Gross Sales',
                        'Sales channel':'Channel',
                        'Total sales':'Sales',
                        'Net sales':'Net Sales'},
                       
                       inplace = True
                       )
        
        shopify['Channel']='Shopify'
        shopify['Wholesale Price']='NaN'
        
        shopify['Order Date'] = pd.to_datetime(shopify['Order Date'],
                                               format="%Y-%m-%d")
        
        shopify_dates=[]
        for dt in list(shopify['Order Date']):
            shopify_dates.append(dt.date())
            
        shopify['Order Date'] = shopify_dates
        
        shopify = shopify[["Order ID",
                           "Channel",
                           "Order Date",
                           "City",
                           "Country",
                           "Product Name",
                           "SKU",
                           "Quantity",
                           "Wholesale Price",
                           "Gross Sales",
                           "Discounts",
                           "Net Sales",
                           "Sales",
                           "Returns",
                           "Shipping",
                           "Taxes"]]
        
        product_tagging(shopify)    
        
        
        
        
        
        
        
        
        #------------------------------------------------------ANKORSTORE------------------------------------------------------
    if ankorstore_upload is not None:
        ankorstore.drop(columns=["Retailer Name",
                                 "Address",
                                 "Zip Code",
                                 "Ship Date",
                                 "status"],
                        
                        inplace=True                
            )
        
        ankorstore.rename(columns = {'Order Number':'Order ID',
                                     'Retail Price':'Sales'},
                          
                          inplace = True
                          )
        
        ankorstore['Channel']='Ankorstore'
        ankorstore['Gross Sales']='NaN'
        ankorstore['Discounts']='NaN'
        ankorstore['Returns']=0
        ankorstore['Shipping']='NaN'
        ankorstore['Taxes']='NaN'
        ankorstore['Net Sales']=ankorstore['Sales']
        
        remove_eurosign(ankorstore,'Wholesale Price')
        remove_eurosign(ankorstore,'Net Sales')
        remove_eurosign(ankorstore,'Sales')
        
        ankorstore['Order Date'] = pd.to_datetime(ankorstore['Order Date'],
                                                  format="%d-%m-%Y")
        
        ankorstore_dates=[]
        for dt in list(ankorstore['Order Date']):
            ankorstore_dates.append(dt.date())
            
        ankorstore['Order Date'] = ankorstore_dates
        
        ankorstore = ankorstore[["Order ID",
                                 "Channel",
                                 "Order Date",
                                 "City",
                                 "Country",
                                 "Product Name",
                                 "SKU",
                                 "Quantity",
                                 "Wholesale Price",
                                 "Gross Sales",
                                 "Discounts",
                                 "Net Sales",
                                 "Sales",
                                 "Returns",
                                 "Shipping",
                                 "Taxes"]]
        
        product_tagging(ankorstore)
        
        
        
        
        
        
        
        #------------------------------------------------------AVOCADO STORE------------------------------------------------------
    if avocadostore_upload is not None:
        avocado.drop(columns = ["eingegangen",
                                "Gesamtbetrag",
                                "Kunde_Name",
                                "Status",
                                "Retouren_Status",
                                "Rechnungsadresse",
                                "Versandadresse",
                                "Kundennummer"],
                     
                        inplace=True                
            )
        
        avocado.rename(columns = {'Order': 'Order ID',
                                  'bezahlt': 'Order Date',
                                  'Produkte': 'Product Name',
                                  'Produkte_SKUs': 'SKU',
                                  'Warenwert': 'Sales',
                                  'Versandkosten': 'Shipping',
                                  'Kunde_Stadt': 'City',
                                  'Retourenwarenwert': 'Returns'},
                       inplace=True
                       )
        
        avocado['Channel']='Avocadostore'
        avocado['Gross Sales']='NaN'
        avocado['Discounts']='NaN'
        avocado['Taxes']='NaN'
        avocado['Net Sales']=avocado['Sales']
        avocado['Wholesale Price']='NaN'
        avocado['Quantity']=1
        avocado['Country']='NaN'
        
        get_country(avocado)
        
        
        avocado['Order Date'] = pd.to_datetime(avocado['Order Date'],
                                               format="%d.%m.%Y")
        
        avocado_dates=[]
        for dt in list(avocado['Order Date']):
            avocado_dates.append(dt.date())
            
        avocado['Order Date'] = avocado_dates
        
        char_stand(avocado)
        
        
        avocado = avocado[["Order ID",
                           "Channel",
                           "Order Date",
                           "City",
                           "Country",
                           "Product Name",
                           "SKU",
                           "Quantity",
                           "Wholesale Price",
                           "Gross Sales",
                           "Discounts",
                           "Net Sales",
                           "Sales",
                           "Returns",
                           "Shipping",
                           "Taxes"]]
        
        
        product_tagging_avocado(avocado)
        
        
        
        
        
        
        
        
        
        
        
        
        
         
    
    #---------------------------------------------Joining standardized data sources in one dataframe-----------------------------------------------------
    
    Orders = pd.concat([ankorstore, shopify, avocado])
    
    Orders.sort_values('Order Date', inplace=True)
    
    Orders.reset_index(inplace=True, drop=True)
    
    
    
    #Creating product kind dummy variables to count units sold
#    Orders['mirage']=0
#    Orders['save the wildlife']=0
#    Orders['save the plants']=0
#    Orders['cloudcastle']=0
#    Orders['granito']=0
#    Orders['an apple a day']=0
#    Orders['save the gardens']=0
#    Orders['save the jungle']=0
    
    
#    def get_prod_var(df):
#        numbers = []
#        
#        for items in df['SKU']:
#            numbers.append(len(items))
#            
#        df['Number Designs'] = numbers
#        
#        return "Done!"
#            
#    get_prod_var(Orders)
    
    
    
    #Getting quantities
    
    def get_quantity(df):
        total=[]    
        for i, row in df.iterrows():
            if row['Channel']=="Shopify":
                if "Big Steps" in row['Product Name']:
                    total.append(row['SKU'][0])
                    total.append(row['SKU'][1])
                if "Three Of A Kind" in row['Product Name']:
                    total.extend([row['SKU'][0],row['SKU'][0],row['SKU'][0]])                
                elif len(row['SKU'])==1:
                    total.append(row['SKU'][0])
            if row['Channel']=="Ankorstore":
                if "Set of 3" in row['Product Name']:
                    total.extend([row['SKU'][0],row['SKU'][0],row['SKU'][0]])
                elif len(row['SKU'])==1:
                    total.append(row['SKU'][0])
            if row['Channel']=="Avocadostore":
                if "1 x 3er" in row['Product Name']:
                    total.extend([row['SKU'][0],row['SKU'][0],row['SKU'][0]])
                if "2 x 3er" in row['Product Name']:
                    if len(row['SKU'])==1:
                        total.extend([row['SKU'][0],row['SKU'][0],row['SKU'][0],row['SKU'][0],row['SKU'][0],row['SKU'][0]])
                    else:
                        total.extend([row['SKU'][0],row['SKU'][0],row['SKU'][0],row['SKU'][1],row['SKU'][1],row['SKU'][1]])
                if "3 x 3er" in row['Product Name']:
                    if len(row['SKU'])==1:
                        total.extend([row['SKU'][0],row['SKU'][0],row['SKU'][0],row['SKU'][0],row['SKU'][0],row['SKU'][0],row['SKU'][0],row['SKU'][0],row['SKU'][0]])
                    elif len(row['SKU'])==2:
                        total.extend([row['SKU'][0],row['SKU'][0],row['SKU'][0],row['SKU'][0],row['SKU'][0],row['SKU'][0],row['SKU'][1],row['SKU'][1],row['SKU'][1]])
                    elif len(row['SKU'])==3:
                        total.extend([row['SKU'][0],row['SKU'][0],row['SKU'][0],row['SKU'][2],row['SKU'][2],row['SKU'][2],row['SKU'][1],row['SKU'][1],row['SKU'][1]])
                    
        return(total)
                        
                        
    get_quantity(Orders)            
    
    
    prod_dict=Counter(get_quantity(Orders))
    
    
    
    
    
    
    
    
    
    
    
    
    #DATA IMPORT
    data_full = Orders
    
    #visits_full = pd.read_csv('G:/My Drive/Fred/4. Education/Ramon Llull University, ESADE/Academic/Courses/Final project/The Sustainables/Shopify Data - Visits.csv')
    
    
    
    
    if shopify_upload is not None or ankorstore_upload is not None or avocadostore_upload is not None:
    
        
        #DATA STANDARDIZATION FOR DATA FULL
        data_full['Order ID'] = data_full['Order ID'].apply(str)
        
        try:
            data_full['Order Date'] = data_full['Order Date'].dt.strftime('%Y-%m-%d')
        except:
            data_full['Order Date'] = data_full['Order Date']
        
        data_full['SKU'] = data_full['SKU'].apply(str)
        data_full['Net Sales'] = data_full['Net Sales'].apply(float)
        data_full['Sales'] = data_full['Sales'].apply(float)
        data_full['Wholesale Price'] = data_full['Wholesale Price'].apply(float)
        data_full['Gross Sales'] = data_full['Gross Sales'].apply(float)
        data_full['Discounts'] = data_full['Discounts'].apply(float)
        data_full['Shipping'] = data_full['Shipping'].apply(float)
        data_full['Taxes'] = data_full['Taxes'].apply(float)
        
        
        
        
        
        
        
        
        
        
        
            
        #DATA PREP
        data = data_full.drop(columns = ['Product Name','Wholesale Price', 'Shipping', 'Taxes'])
        
        
        dates_orders = pd.to_datetime(data["Order Date"]).dt.date
        min_order_date =  dates_orders.min()
        max_order_date = dates_orders.max()
        
        if visits_full is not None:
            dates_visits = pd.to_datetime(visits_full["day"]).dt.date
            min_visit_date = dates_visits.min()
            max_visit_date = dates_visits.max()
            
            
            visits_full["day"]=dates_visits
            
            
            if min_order_date < min_visit_date:
                min_date = min_order_date
            else:
                min_date = min_visit_date
            
        
            if max_order_date < max_visit_date:
                max_date = max_visit_date
            else:
                max_date = max_order_date
        else:
            max_date=max_order_date
            min_date=min_order_date
            
        
        if visits_full is not None:
            
            visit_country = visits_full['location_country']
            
            for i, visit in enumerate(visit_country):
                if visit is None:
                    visit_country[i]=str("")
                else:
                    visit_country[i]=str(visit)
            
            visits_full['location_country'] = visit_country
            
        
        
        prods = ['mirage',
                 'save the wildlife',
                 'save the plants',
                 'cloudcastle',
                 'granito',
                 'an apple a day',
                 'save the gardens',
                 'save the jungle'
                 ]
        
        
        
        
        
        
        
        
        
        
            
        #SIDEBAR & KPI FILTERS
        
        st.sidebar.subheader("Preferences")
        
        
        
        
        date_sample = st.sidebar.date_input("Select date range",[], min_value = min_date, max_value = max_date)
        
        try:
            selected_data = data[data['Order Date'] <= date_sample[1]]
            if visits_full is not None:
                selected_visits = visits_full[visits_full['day'] <= date_sample[1]]
            max_date = date_sample[1]
        except:
            selected_data = data[data['Order Date'] <= max_date]
            if visits_full is not None:
                selected_visits = visits_full[visits_full['day'] <= max_date]
            
        try:
            selected_data = selected_data[selected_data['Order Date'] >= date_sample[0]]
            if visits_full is not None:
                selected_visits = selected_visits[selected_visits['day'] >= date_sample[0]]
            min_date = date_sample[0]
        except:
            selected_data = selected_data[selected_data['Order Date'] >= min_date]
            if visits_full is not None:
                selected_visits = selected_visits[selected_visits['day'] >= min_date]
                    
    
    #LAYOUT    
        st.subheader("Showing data from "+str(min_date)+" to "+str(max_date))
        
        st.title(" ")
    
    
        
        orderstatsfilter = st.sidebar.selectbox("Would you like to filter the key Orders statistics?", ("Yes", "No"))
        
        visitstatsfilter = st.sidebar.selectbox("Would you like to filter the key Visits statistics?", ("Yes", "No"))
        
        ordergraphfilter = st.sidebar.selectbox("Would you like to filter the Orders graphs?", ("Yes", "No"))
        
        visitgraphfilter = st.sidebar.selectbox("Would you like to filter the Visits graphs?", ("Yes", "No"))
        
        
        product_array = np.array(prods)
        product_array = np.insert(product_array,0, np.array(""))
        gfilter1 = st.sidebar.selectbox(
            'Select Product',
             product_array,
             key = "gfilter1")
        
        #select Channel
        channel_array = data['Channel'].unique()
        channel_array = np.insert(channel_array, 0, np.array(""))
        gfilter2 = st.sidebar.selectbox(
            'Select Channel',
             channel_array,
             key = "gfilter2")
        
        #select Country
        if visits_full is not None:
            country_list = visits_full['location_country'].unique()
        else:
            country_list = Orders['Country'].unique()
        country_list = np.insert(country_list,0, np.array(""))
        gfilter3 = st.sidebar.selectbox(
                'Select Country',
                 country_list,
                 key = "gfilter3")
            
        #Select Browser
        if visits_full is not None:
            browser_list = visits_full['ua_browser'].unique()
            browser_list = np.insert(browser_list, 0, np.array(""))
            gfilter4 = st.sidebar.selectbox(
                'Select Browser',
                browser_list,
                key = 'gfilter4'
                )
            
        #Select Device
        if visits_full is not None:
            device_list = visits_full['ua_os'].unique()
            device_list = np.insert(device_list, 0, np.array(""))
            gfilter5 = st.sidebar.selectbox(
                'Select Device',
                device_list,
                key = 'gfilter5'
                )
            
        #Select Promotion/Non-promotion
        
        
        
        
        stats_data = selected_data
        
        if gfilter1 is not None:
            stats_data = stats_data[stats_data['SKU'].str.contains(gfilter1)]
        if gfilter2 is not None:
            stats_data = stats_data[stats_data['Channel'].str.contains(gfilter2)]
        if gfilter3 is not None:
            stats_data = stats_data[stats_data['Country'].str.contains(gfilter3)]    
        
        
        if orderstatsfilter == "No":
            stats_data = selected_data
        
        
        graph_data = selected_data
        
        if gfilter1 is not None:
            graph_data = graph_data[graph_data['SKU'].str.contains(gfilter1)]
        if gfilter2 is not None:
            graph_data = graph_data[graph_data['Channel'].str.contains(gfilter2)]
        if gfilter3 is not None:
            graph_data = graph_data[graph_data['Country'].str.contains(gfilter3)]    
            
        
        if ordergraphfilter=="No":
            graph_data = selected_data
         
        if visits_full is not None:
            visitsstats_data = selected_visits
            
            if gfilter3 is not None:
                visitsstats_data = visitsstats_data[visitsstats_data['location_country'].str.contains(gfilter3)]
            if gfilter4 is not None:
                visitsstats_data = visitsstats_data[visitsstats_data['ua_browser'].str.contains(gfilter4)]
            if gfilter5 is not None:
                visitsstats_data = visitsstats_data[visitsstats_data['ua_os'].str.contains(gfilter5)]
            
            if visitstatsfilter=="No":
                visitsstats_data = selected_visits
                
            
                
            visitsgraph_data = selected_visits
            
            if gfilter3 is not None:
                visitsgraph_data = visitsgraph_data[visitsgraph_data['location_country'].str.contains(gfilter3)]
            if gfilter4 is not None:
                visitsgraph_data = visitsgraph_data[visitsgraph_data['ua_browser'].str.contains(gfilter4)]
            if gfilter5 is not None:
                visitsgraph_data = visitsgraph_data[visitsgraph_data['ua_os'].str.contains(gfilter5)]
            
            if visitgraphfilter=="No":
                visitsgraph_data = selected_visits
                
        
        
        
        
        
        
        
        
        
        
        
        
        
            
        #DATA PREP
        
            
        #getting distinct products
        
        all_prods = graph_data['SKU']
        
        new_prods = []
        for i in all_prods:
            i1 = i.replace("[", "")
            i2 = i1.replace("]", "")
            i3 = i2.split(', ')
            new_prods = new_prods + i3
            
        
        #getting avg visit times
        if visits_full is not None:
                
            visit_times=list(visitsstats_data["avg_duration"])
            
            for i,visit in enumerate(visit_times):
                try:
                    hours = round(float(visit[:visit.index("h")]),0)
                    minutes = round(float(visit[visit.index("h")+1:visit.index("m")]),0)
                    seconds = round(float(visit[visit.index("m")+1:visit.index("s")]),0)
                except:
                    try:
                        hours = 0
                        minutes = round(float(visit[:visit.index("m")]),0)
                        seconds = round(float(visit[visit.index("m")+1:visit.index("s")]),0)
                    except:
                        hours = 0
                        minutes = 0
                        seconds = round(float(visit[:visit.index("s")]),0)
                visit_times[i] = int(hours*3600) + int(minutes*60) + int(seconds)
                        
            visitsstats_data["avg_duration"]=visit_times
            
        
        
        
        
        
        
        
        
        
        
        
            
        #STATS
        st.header("Key Stats")
        st.text(" ")
        
        #STATS DATA
        total_revenue = stats_data["Sales"].sum()
        n_orders = stats_data["Sales"].count()
        if visits_full is not None:
            
            total_visitors = visitsstats_data["total_visitors"].sum()
            conversion_rate = visitsstats_data["total_conversion"].sum() / visitsstats_data.shape[0]
        
        n_returns = stats_data['Returns'].sum()
        avg_basket = stats_data['Sales'].mean()
        if visits_full is not None:
                
            avg_duration = visitsstats_data["avg_duration"].mean()
            avg_pageviews = visitsstats_data["total_pageviews"].mean()
        
        
        
        
            
        #STATS LAYOUT
        c1, c2, c3, c4 = st.columns(4)
        
        c1.metric("Total Revenue", str(round(total_revenue,2))+"€")
        
        c2.metric("Number of Orders", str(n_orders))
        
        if visits_full is not None:
            c3.metric("Number of Visits", str(total_visitors))
            
            c4.metric("Shopify Conversion Rate", str(round(conversion_rate,3)*100)+"%")
        
        
        st.write(" ")
        
        
        l2c1, l2c2, l2c3, l2c4 = st.columns(4)
        
        l2c1.metric("Number of Returns", int(n_returns))
        
        l2c2.metric("Average Basket Size", str(round(avg_basket,2))+"€")
        
        if visits_full is not None:
            l2c3.metric("Average Visit Duration", str(round(avg_duration,2))+" sec")
            
            l2c4.metric("Average Page Views", str(round(avg_pageviews,2)))
        
        
        
        
        
            
        #GRAPHS
        
        st.write(" ")
        st.header("Graphs")
        
        #GRAPH DATA
        
        #1 - timeline of orders
        timeline = graph_data[["Order Date"]]
        df = timeline.groupby('Order Date').size().values
        timeline = timeline.drop_duplicates(subset="Order Date").assign(Count=df)
        timeline = timeline.set_index('Order Date')
        
        #2- number of orders per product
        
        #getting distinct categories
        
        #getting count per category    
        values, counts = np.unique(new_prods, return_counts=True)
        #dataframe with data
        df_data = {"Product Name":list(prod_dict.keys()), "Count":list(prod_dict.values())}
        df2 = pd.DataFrame(data = list(prod_dict.values()), index = list(prod_dict.keys()))
        
        #df2 = pd.DataFrame(data=counts, index=values)
        
        #3 - stats by channel
        ord_by_chan = graph_data[['Channel','Sales']].groupby("Channel").count()
        
        #4-Revenue over time
        timeline2 = graph_data[['Order Date', 'Sales']].groupby("Order Date").sum()
        
        #5-Revenue per channel
        rev_chan = graph_data[['Channel','Sales']].groupby('Channel').sum()
        
        #-------------------------------------------------------------
        
        if visits_full is not None:
            #6-Visits over time
            timeline3 = visitsgraph_data[['day','total_visitors']]
            df3 = timeline3.groupby('day').sum()
            timeline3 = timeline3.set_index('day').groupby('day').sum()
            
            #7-Conversions over time
            timeline4 = visitsgraph_data[['day','total_conversion']]
            df4 = timeline4.groupby('day').sum()
            timeline4 = timeline4.set_index('day').groupby('day').sum()
            
            #8-Browser Conversion
            browser_conv = visitsgraph_data[['ua_browser', 'total_conversion', 'total_visitors']].groupby('ua_browser').sum()
            conv_by_browser = pd.DataFrame()
            conv_by_browser['Conversion (%)'] = browser_conv['total_conversion']/browser_conv['total_visitors']*100
            conv_by_browser['Total Conversions'] = browser_conv['total_conversion']
            conv_by_browser = conv_by_browser.sort_values('Conversion (%)',ascending=False).head(10)
            
            #9-Device Conversion
            device_conv = visitsgraph_data[['ua_os', 'total_conversion', 'total_visitors']].groupby('ua_os').sum()
            conv_by_device = pd.DataFrame()
            conv_by_device['Conversion (%)'] = device_conv['total_conversion']/device_conv['total_visitors']*100
            conv_by_device['Total Conversions'] = device_conv['total_conversion']
            conv_by_device = conv_by_device.sort_values('Conversion (%)',ascending=False)
            
            #10-Visits by country
            country_visits = visitsgraph_data[['location_country','total_visitors','total_conversion']].groupby('location_country').sum().sort_values('total_visitors',ascending=False).head(10)
            country_visits['Conversion (%)'] = country_visits['total_conversion'] / country_visits['total_visitors']*100
            #11- Conversion by promotion medium
        
        
        
        
        
        
        
            
        #GRAPH LAYOUT
        st.title("Orders")
        
        
        #Row 1
        row1col1, row1col2 = st.columns(2)
        
        #1
        row1col1.header("Orders Timeline")
        row1col1.line_chart(data = timeline, width = 500, height =300)
        
        #2
        row1col2.header("Number of orders per channel")
        row1col2.bar_chart(ord_by_chan, width = 500, height =300)
        
        
        #Row 2
        row2col1, row2col2 = st.columns(2)
        
        #3
        row2col1.header("Sales Timeline")
        row2col1.line_chart(data=timeline2, width = 500, height =300)
        
        #4
        row2col2.header("Revenue by Channel")
        row2col2.bar_chart(data=rev_chan, width = 500, height =300)
        
        
        #Row 3
        
        st.header("Number of orders per Product")
        st.bar_chart(data = df2, width = 500, height =300)
        
        
        
        
        
        
        if visits_full is not None:
            st.title("Visits")
            
            #Row 4
            row4col1, row4col2 = st.columns(2)
            
            #6-Visits over time
            row4col1.header("Visits Timeline")
            row4col1.line_chart(data=timeline3, width = 500, height =300)
            
            #7
            row4col2.header("Conversions Timeline")
            row4col2.line_chart(data=timeline4, width = 500, height =300)
            
            #Row 5
            row5col1, row5col2 = st.columns(2)
            
            #8-Conversions by browser
            row5col1.header("Conversions by Browser")
            row5col1.table(conv_by_browser)
            
            #9-Conversions by device
            row5col2.header("Conversions by Device")
            row5col2.table(conv_by_device)
            
            #Row 6
            
            #10
            st.header("Visits by country (Top 10)")
            st.table(country_visits)
            
            #11
        
        
        
        
        st.title(" ")
        
        
        
        
        
        
        
        #ALLDATA
        
        st.title("All Orders")
        
        
        #sorting = st.selectbox("Sort by", (None, "Order Date", "Sales"))
        
        sorted_data = Orders
            
        
        
        #if sorting == "Order Date":
        #  sorted_data = sorted_data.sort_values(by='Order Date', ascending = False)
        #if sorting == "Sales":
        #  sorted_data = sorted_data.sort_values(by='Sales', ascending = False)
    
    
    
    
    
    
        
    #show filtered data
        if st.checkbox('Show filtered data'):
            st.dataframe(sorted_data)
    else:
        st.title("Please upload files to proceed")
