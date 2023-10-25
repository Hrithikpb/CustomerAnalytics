from flask import Flask, request, render_template,send_file, url_for, redirect
from itsdangerous import json
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import io
import json
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sqlalchemy import create_engine
import plotly
import plotly.express as px
import calendar
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from flask_mail import Mail, Message
import smtplib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

##########################################################################################
# File upload
@app.route('/fileupload')
def fileupload():
    return render_template('fileupload.html')

# Features
@app.route('/features', methods = ['GET','POST'])
def features():
    if request.method == 'POST' :

# ----------------------------------------------------------------------------------------

# # Services
# @app.route('/services')
# def services():
#     return render_template('services.html')

# # About us
# @app.route('/about_us')
# def about_us():
#     return render_template('about_us.html')

# ----------------------------------------------------------------------------------------
        # saving file
        f = request.files['myfile']  
        f.filename = 'dataset.csv'
        f.save(f.filename)
    return render_template('features_page.html')

#########################################################################################
# Basic Insights

@app.route('/features/basicinsights', methods = ['GET', 'POST'])
def basicinsights():

    if request.method == 'POST':
        dataset = pd.read_csv('dataset.csv')
        # Checking null values
        null = dataset.isnull().values.any() 
        dimension = dataset.shape      # dimension of dataset
        columns = list(dataset.columns) # features of dataset
        null_df = pd.DataFrame(dataset.isnull().sum()).reset_index()
        null_df.columns = ['Columns', 'No. of missing values']
        fig1 = px.bar(null_df, x = 'Columns', y = 'No. of missing values', title = "Missing values"  )
        graph1 = fig1.to_html(full_html = False, include_plotlyjs = 'cdn')
        top10 = dataset[['PRODUCT NAME', 'MRP']].drop_duplicates(keep='first').sort_values(by = 'MRP', ascending = False).reset_index().drop(['index'], axis =1).head(10)
        bottom10 = dataset[['PRODUCT NAME', 'MRP']].drop_duplicates(keep='first').sort_values(by = 'MRP', ascending = True).reset_index().drop(['index'], axis =1).head(10)
        d1 = pd.DataFrame(dataset[['PRODUCT NAME', 'MRP']].drop_duplicates(keep='first').sort_values(by = 'MRP', ascending = False).reset_index().drop(['index'], axis =1)['PRODUCT NAME'].value_counts()).reset_index() 
        d1.columns = ['PRODUCT NAME', 'Price changes']
        price_changed_items = list(d1[d1['Price changes']> 1]['PRODUCT NAME'])

        selected_product= request.form["products"]
        selected_product = str(selected_product)
        selected_product_df = dataset[dataset['PRODUCT NAME']== selected_product][['DATE', 'MRP']].drop_duplicates(keep ='first')
        fig2 = px.line(selected_product_df, x = 'DATE', y = 'MRP', title = "CHANGE IN PRICE"  )
        graph2 = fig2.to_html(full_html = False, include_plotlyjs = 'cdn')
        top_10_spending_customers = dataset[['AMOUNT', 'CUSTOMER REFERENCE']].groupby(by=['CUSTOMER REFERENCE']).sum().sort_values(by = ['AMOUNT'], ascending = False).reset_index().head(10)
        fig3 = px.bar(dataset[['AMOUNT', 'CUSTOMER REFERENCE']].groupby(by=['CUSTOMER REFERENCE']).sum().reset_index(), x = 'CUSTOMER REFERENCE', y = 'AMOUNT', title = "SPENDING NATURE" )
        graph3 = fig3.to_html(full_html = False, include_plotlyjs = 'cdn')
        return render_template('basic_insights.html', table1 = [null_df.to_html()], table2 = [top10.to_html()], table3 = [bottom10.to_html()], table4 = [top_10_spending_customers.to_html()], graph1 = graph1, graph2 = graph2, graph3 = graph3, dimension=dimension, columns = columns, null = null, dataset= dataset, top10=top10, bottom10 = bottom10, price_changed_items = price_changed_items )
    
    else:
        dataset = pd.read_csv('dataset.csv')
        # Checking null values
        null = dataset.isnull().values.any() 
        dimension = dataset.shape      # dimension of dataset
        columns = list(dataset.columns) # features of dataset
        null_df = pd.DataFrame(dataset.isnull().sum()).reset_index()
        null_df.columns = ['Columns', 'No. of missing values']
        fig1 = px.bar(null_df, x = 'Columns', y = 'No. of missing values', title = "Missing values"  )
        graph1 = fig1.to_html(full_html = False, include_plotlyjs = 'cdn')
        top10 = dataset[['PRODUCT NAME', 'MRP']].drop_duplicates(keep='first').sort_values(by = 'MRP', ascending = False).reset_index().drop(['index'], axis =1).head(10)
        bottom10 = dataset[['PRODUCT NAME', 'MRP']].drop_duplicates(keep='first').sort_values(by = 'MRP', ascending = True).reset_index().drop(['index'], axis =1).head(10)
        d1 = pd.DataFrame(dataset[['PRODUCT NAME', 'MRP']].drop_duplicates(keep='first').sort_values(by = 'MRP', ascending = False).reset_index().drop(['index'], axis =1)['PRODUCT NAME'].value_counts()).reset_index() 
        d1.columns = ['PRODUCT NAME', 'Price changes']
        price_changed_items = list(d1[d1['Price changes']> 1]['PRODUCT NAME'])
        top_10_spending_customers = dataset[['AMOUNT', 'CUSTOMER REFERENCE']].groupby(by=['CUSTOMER REFERENCE']).sum().sort_values(by = ['AMOUNT'], ascending = False).reset_index().head(10)
        fig3 = px.bar(dataset[['AMOUNT', 'CUSTOMER REFERENCE']].groupby(by=['CUSTOMER REFERENCE']).sum().reset_index(), x = 'CUSTOMER REFERENCE', y = 'AMOUNT', title = "SPENDING NATURE" )
        graph3 = fig3.to_html(full_html = False, include_plotlyjs = 'cdn')
        return render_template('basic_insights.html', table1 = [null_df.to_html()], table2 = [top10.to_html()], table3 = [bottom10.to_html()], table4 = [top_10_spending_customers.to_html()] , graph1 = graph1, graph3 = graph3, dimension=dimension, columns = columns, null = null, dataset= dataset, top10=top10, bottom10 = bottom10, price_changed_items = price_changed_items )











##########################################################################################
# MARKET BASKET ANALYSIS 

#Zhangs Function
def zhang(antecedent, consequent):
    supportA = antecedent.mean()
    supportC = consequent.mean()
    supportAC = np.logical_and(antecedent, consequent).mean()
    numerator = supportAC - supportA*supportC
    denominator = max(supportAC*(1-supportA), supportA*(supportC-supportAC))
    zhang = numerator / denominator
    return zhang 

def listtostrings(l):
    if len(l) == 1:
        return str(l[0])
    else:
        return str(l[0] + ' |||||| ' + l[1]) 

# ----------------------------------------------------------------------------------------
@app.route('/features/market-basket-analysis')
def marketbasketanalysis():
# ----------------------------------------------------------------------------------------
    # reading dataset
    dataset = pd.read_csv('dataset.csv')
    dataset = dataset.dropna()
# ----------------------------------------------------------------------------------------
    # total transactions, total customers, total products sold 
    total_transactions = dataset['VOUCHERNO'].nunique()        # total transactions 
    total_customers = dataset['CUSTOMER REFERENCE'].nunique()  # total customers
    total_products = dataset['PRODUCT NAME'].nunique()         # total products sold
    list_of_top_10_items = []                                  # top 10 items
    for i in dict(dataset['PRODUCT NAME'].value_counts().head(10)).items():  
        list_of_top_10_items.append([i[0], i[1]])
# ----------------------------------------------------------------------------------------
    #building transactions
    transactions = dataset[['VOUCHERNO', 'PRODUCT NAME']]
    transactions = transactions.groupby('VOUCHERNO')['PRODUCT NAME'].apply(list).reset_index(name="PRODUCT NAME")
    transactions = transactions.rename(columns = {'VOUCHERNO' : 'TID', 'PRODUCT NAME': 'Transaction'})
    transaction_list = list(transactions['Transaction'])
# ----------------------------------------------------------------------------------------
    #one-hot encoding
    encoder = TransactionEncoder().fit(transaction_list)
    onehot = encoder.transform(transaction_list)
    onehot = pd.DataFrame(onehot, columns = encoder.columns_)
# ----------------------------------------------------------------------------------------
    #ARM FP growth 0.4% support and 35% confidence
    frequent_itemsets = fpgrowth(onehot, min_support = 0.004, use_colnames =True)
    rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.35)
    list_of_association_rules = []
    for i in range(len(rules)):
        list_of_association_rules.append([list(rules['antecedents'][i]), list(rules['consequents'][i])])
# ----------------------------------------------------------------------------------------
    #Zhangs analysis
    t_list2D = []
    for i in transaction_list:
        if len(i) == 2:
            t_list2D.append(i)         

    zhangs_metric = []

    for i in t_list2D:
        antecedent = onehot[i[0]]
        consequent = onehot[i[1]]
        zhangs_metric.append([i[0],i[1],zhang(antecedent, consequent)])

    zhangs_score = pd.DataFrame(zhangs_metric, columns =['antecedents', 'consequents', "zhangs score"])

    # Poor Association
    poor_zhangs = zhangs_score[zhangs_score['zhangs score'] < 0].reset_index().drop('index', axis = 1) 
    list_of_poor_association_rules = []
    for i in range(len(poor_zhangs)):
        list_of_poor_association_rules.append([poor_zhangs['antecedents'][i], poor_zhangs['consequents'][i]])
# -----------------------------------------------------------------------------------------
    
    frequency_of_top10_items = pd.DataFrame(list_of_top_10_items)
    frequency_of_top10_items.columns = ['PRODUCT NAME', 'FREQUENCY']
    fig1 = px.bar(frequency_of_top10_items, x = 'PRODUCT NAME', y = 'FREQUENCY', title = "TOP 10 MOST FREQUENT PRODUCTS" )
    graph1 = fig1.to_html(full_html = False, include_plotlyjs = 'cdn')

    rules2 = rules
    rules2['antecedents'] = rules2['antecedents'].apply(lambda x: list(x))
    rules2['consequents'] = rules2['consequents'].apply(lambda x: list(x))
    rules2['antecedents'] = rules2['antecedents'].apply(listtostrings)
    rules2['consequents'] = rules2['consequents'].apply(listtostrings)
    rules2 = rules2.sort_values(by = ['confidence'], ascending = False)
    rules2 = rules2[['antecedents', 'consequents']]
    rules2.to_csv('High Association Products.csv', encoding = 'utf-8-sig', index = False)

    return render_template('market-basket-analysis.html', 
        total_transactions = total_transactions,
        total_customers = total_customers,
        total_products = total_products,
        list_of_top_10_items =list_of_top_10_items, 
        list_of_association_rules = list_of_association_rules,
        list_of_poor_association_rules = list_of_poor_association_rules,
        frequency_of_top10_items = frequency_of_top10_items,
        graph1 = graph1
        )

@app.route('/features/market-basket-analysis/download-products-file')
def download_high_association_products_file():
    return send_file('High Association Products.csv', as_attachment= True)


#############################################################################################
#############################################################################################

#CUSTOMER PROFILING

#customer profiling function
def customer_profile(custid, dataset):
    cust_dataset = dataset[dataset['CUSTOMER REFERENCE'] == custid]
    dates_of_purchase = cust_dataset['DATE'].unique()
    invoices = cust_dataset['VOUCHERNO'].unique()
    return [dates_of_purchase, invoices]

@app.route('/features/customerid')
def getcustomerids():     
    # reading dataset
    dataset = pd.read_csv('dataset.csv')
    dataset = dataset.dropna()    
    customer_ids = sorted(list(dataset['CUSTOMER REFERENCE'].unique()))
    return render_template('cust_prof_cust_ids.html',customer_ids = customer_ids)


def RScore(x,p,d):
    if x <= d[p][0.20]:
        return 1
    elif x <= d[p][0.40]:
        return 2
    elif x <= d[p][0.60]: 
        return 3
    elif x <= d[p][0.80]:
        return 4
    else:
        return 5
    
def FScore(x,p,d):
    if x <= d[p][0.20]:
        return 5
    elif x <= d[p][0.40]:
        return 4
    elif x <= d[p][0.60]: 
        return 3
    elif x <= d[p][0.80]: 
        return 2
    else:
        return 1
    
def MScore(x,p,d):
    if x <= d[p][0.20]:
        return 5
    elif x <= d[p][0.40]:
        return 4
    elif x <= d[p][0.60]: 
        return 3
    elif x <= d[p][0.80]: 
        return 2
    else:
        return 1

@app.route('/features/customerid/customerprofile', methods=['POST'])
def customerprofile():
    if request.method == 'POST':
        dataset = pd.read_csv('dataset.csv')

        customer_data = []
        for i in sorted(dataset['CUSTOMER REFERENCE'].unique()):
            customer_data.append(customer_profile(i, dataset))
        custids = dict(zip(sorted(dataset['CUSTOMER REFERENCE'].unique()), customer_data))

        custID= request.form["custids"]
        custID = int(custID)
        desired_customer = custids[custID] 
        dates_of_purchase = desired_customer[0]                      # dates of purchase
        invoices = desired_customer[1]                               # invoices
        number_of_transactions= len(invoices)                        # no. of invoices 
        customer_transaction_table = dataset[dataset['CUSTOMER REFERENCE'] == custID].reset_index().drop(['index', 'CUSTOMER REFERENCE'], axis = 1)
        pf = dataset[dataset['CUSTOMER REFERENCE'] == custID].reset_index().drop(['index'], axis = 1)[['PRODUCT NAME', 'QTY']]
        product_freq = pf.groupby(['PRODUCT NAME']).sum().sort_values(by =['QTY'], ascending =False).reset_index()
        number_of_products = len(pf['PRODUCT NAME'].unique())
        cp = dataset[dataset['CUSTOMER REFERENCE'] == custID].reset_index().drop(['index'], axis = 1)
        cp['DATE'] = pd.to_datetime(cp['DATE'])
        cp['MONTH'] = cp['DATE'].dt.month
        cp['YEAR'] = cp['DATE'].dt.year
        cp['MONTH_YEAR'] = cp['MONTH'].astype('str')  + '-' + cp['YEAR'].astype('str')
        cp['MONTH_YEAR'] = pd.to_datetime(cp['MONTH_YEAR'])
        monthly_frequency = cp[['VOUCHERNO', 'MONTH_YEAR']].groupby(by = ['VOUCHERNO', 'MONTH_YEAR']).sum().reset_index().drop('VOUCHERNO', axis = 1)
        monthly_frequency['FREQUENCY'] = 1
        monthly_frequency = monthly_frequency.groupby(['MONTH_YEAR']).sum().reset_index()
        monthly_frequency['MONTH'] = monthly_frequency['MONTH_YEAR'].dt.month
        monthly_frequency['YEAR'] = monthly_frequency['MONTH_YEAR'].dt.year
        monthly_frequency['MONTH'] = monthly_frequency['MONTH'].apply(lambda x: calendar.month_abbr[x])
        monthly_frequency['MONTH_YEAR'] = monthly_frequency['MONTH'] + ' ' + monthly_frequency['YEAR'].astype(str)
        monthly_frequency = monthly_frequency.drop(['MONTH', 'YEAR'], axis= 1)
        total_frequency = monthly_frequency['FREQUENCY'].sum()
        monthly_expenditure = cp[['MONTH_YEAR','AMOUNT']].groupby(by=["MONTH_YEAR"]).sum()
        monthly_expenditure = monthly_expenditure.reset_index().sort_values(by = ['MONTH_YEAR'])
        monthly_expenditure['MONTH'] = monthly_expenditure['MONTH_YEAR'].dt.month
        monthly_expenditure['YEAR'] = monthly_expenditure['MONTH_YEAR'].dt.year
        monthly_expenditure['MONTH'] = monthly_expenditure['MONTH'].apply(lambda x: calendar.month_abbr[x])
        monthly_expenditure['MONTH_YEAR'] = monthly_expenditure['MONTH'] + ' ' + monthly_expenditure['YEAR'].astype(str)
        monthly_expenditure = monthly_expenditure.drop(['MONTH', 'YEAR'], axis= 1)
        average_monthly_expenditure = monthly_expenditure['AMOUNT'].sum()/ monthly_expenditure.shape[0]
        total_expenditure = monthly_expenditure['AMOUNT'].sum()
        data = dataset
        amount = pd.DataFrame(data.QTY * dataset.MRP, columns = ['MONETARY'])
        data_cust = np.array(data['CUSTOMER REFERENCE'], dtype=np.object)
        data_cust = pd.DataFrame(data_cust, columns = ["CUSTOMER REFERENCE"])
        data_cust = pd.concat(objs = [data_cust, amount], axis = 1, ignore_index = False)
        monetary = data_cust.groupby(by = ["CUSTOMER REFERENCE"]).MONETARY.sum()
        monetary = monetary.reset_index()
        monetary = monetary[monetary['CUSTOMER REFERENCE'] != 99999]
        frequency = data[['CUSTOMER REFERENCE', 'VOUCHERNO']]
        frequency_df = frequency.groupby("CUSTOMER REFERENCE").VOUCHERNO.count()
        frequency_df = pd.DataFrame(frequency_df)
        frequency_df = frequency_df.reset_index()
        frequency_df.columns = ["CUSTOMER REFERENCE", "FREQUENCY"]
        data["CUSTOMER REFERENCE"] = data["CUSTOMER REFERENCE"].astype(int) 
        data["DATE"] = pd.to_datetime(data["DATE"])
        today_date = dt.datetime(data["DATE"].max().year,data["DATE"].max().month,data["DATE"].max().day) 
        recency = (today_date - data.groupby("CUSTOMER REFERENCE").agg({"DATE":"max"}))
        recency.rename(columns = {"DATE":"RECENCY"}, inplace = True)
        recency_df = recency["RECENCY"].apply(lambda x: x.days)
        RFMScores = frequency_df.merge(monetary, on = "CUSTOMER REFERENCE")
        RFMScores = RFMScores.merge(recency_df, on = "CUSTOMER REFERENCE")
        quantiles = RFMScores.quantile(q=[0.2,0.4,0.6, 0.8])
        quantiles = quantiles.to_dict()
        RFMScores['R'] = RFMScores['RECENCY'].apply(RScore, args=('RECENCY',quantiles,))
        RFMScores['F'] = RFMScores['FREQUENCY'].apply(FScore, args=('FREQUENCY',quantiles,))
        RFMScores['M'] = RFMScores['MONETARY'].apply(MScore, args=('MONETARY',quantiles,))
        RFMScores['RFM Score'] = RFMScores[['R', 'F', 'M']].sum(axis = 1)
        Loyalty_Level = ['Platinum', 'Gold', 'Silver', 'Bronze', 'Iron']
        Score_cuts = pd.qcut(RFMScores['RFM Score'], q = 5, labels = Loyalty_Level)
        RFMScores['RFM Loyalty Level'] = Score_cuts.values
        RFMScores = RFMScores[['CUSTOMER REFERENCE', 'RFM Loyalty Level']]
        loyalty_level = list(RFMScores[RFMScores['CUSTOMER REFERENCE'] == custID]['RFM Loyalty Level'])

        fig1 = px.bar(product_freq, x = 'PRODUCT NAME', y = 'QTY', title = "PRODUCTS FREQUENCIES" )
        graph1 = fig1.to_html(full_html = False, include_plotlyjs = 'cdn')
        fig2 = px.bar(monthly_frequency, x = 'MONTH_YEAR', y = 'FREQUENCY', title = "MONTHLY FREQUENCY" )
        graph2 = fig2.to_html(full_html = False, include_plotlyjs = 'cdn')
        fig3 = px.bar(monthly_expenditure, x = 'MONTH_YEAR', y = 'AMOUNT', title = "MONTHLY EXPENDITURE" )
        graph3 = fig3.to_html(full_html = False, include_plotlyjs = 'cdn')

# -------------------------------------------------------------------------------------------
    return render_template('customer_profiles.html', 
        custID =custID,
        desired_customer = desired_customer,
        table1 = [customer_transaction_table.to_html()],
        table2 = [product_freq.to_html()],
        table3 = [monthly_frequency.to_html()],
        table4 = [monthly_expenditure.to_html()],
        dates_of_purchase = dates_of_purchase,
        invoices =  invoices,
        number_of_transactions = number_of_transactions,
        number_of_products = number_of_products,
        total_frequency = total_frequency,
        total_expenditure = total_expenditure,
        average_monthly_expenditure = average_monthly_expenditure,
        loyalty_level = loyalty_level,
        graph1 = graph1,
        graph2 = graph2,
        graph3 = graph3
        )





##############################################################################################
##############################################################################################
# Sales
@app.route('/features/sales')
def sales():
    dataset = pd.read_csv('dataset.csv')
    dataset = dataset.dropna()
    # Top 5 Highest Sales Product
    top5salesproducts = dataset[['PRODUCT NAME', 'AMOUNT']].groupby(by='PRODUCT NAME').sum().sort_values(by = 'AMOUNT', ascending= False).head().reset_index()
    # Least 5 Highest Sales Product
    bottom5salesproducts = dataset[['PRODUCT NAME', 'AMOUNT']].groupby(by='PRODUCT NAME').sum().sort_values(by = 'AMOUNT', ascending= True).head().reset_index()
    # Sales per day
    sales_per_day = dataset[['DATE','AMOUNT']]
    sales_per_day['DATE'] = pd.to_datetime(sales_per_day['DATE'], dayfirst = True)
    sales_per_day = sales_per_day.groupby(by = sales_per_day['DATE']).sum()
    sales_per_day = sales_per_day.reset_index()
    fig1 = px.line(sales_per_day.drop_duplicates(keep ='first'), x = 'DATE', y = 'AMOUNT', title = "SALES PER DAY")
    graph1 = fig1.to_html(full_html = False, include_plotlyjs = 'cdn')
    # Highest Sale (Date)
    highest_sale_date = sales_per_day.sort_values(by= ['AMOUNT'], ascending = False).reset_index().drop('index', axis =1).head(1)
    highest_sale_date = highest_sale_date[highest_sale_date['AMOUNT'] == highest_sale_date['AMOUNT'].max()]
    # Lowest Sale (Date)
    lowest_sale_date = sales_per_day.sort_values(by= ['AMOUNT'], ascending = True).reset_index().drop('index', axis =1)
    lowest_sale_date = lowest_sale_date[lowest_sale_date['AMOUNT'] == lowest_sale_date['AMOUNT'].min()]
    # Sales per month
    sales_per_month = dataset[['DATE','AMOUNT']]
    sales_per_month['DATE'] = pd.to_datetime(sales_per_month['DATE'])
    sales_per_month['MONTH'] = sales_per_month['DATE'].dt.month
    sales_per_month['YEAR'] = sales_per_month['DATE'].dt.year
    sales_per_month['MONTH_YEAR'] = sales_per_month['MONTH'].astype('str')  + '-' + sales_per_month['YEAR'].astype('str')
    sales_per_month['MONTH_YEAR'] = pd.to_datetime(sales_per_month['MONTH_YEAR'])
    sales_per_month = sales_per_month[['MONTH_YEAR','AMOUNT']].groupby(by=["MONTH_YEAR"]).sum()
    sales_per_month = sales_per_month.reset_index().sort_values(by = ['MONTH_YEAR'])
    sales_per_month['MONTH'] = sales_per_month['MONTH_YEAR'].dt.month
    sales_per_month['YEAR'] = sales_per_month['MONTH_YEAR'].dt.year
    sales_per_month['MONTH'] = sales_per_month['MONTH'].apply(lambda x: calendar.month_abbr[x])
    sales_per_month['MONTH_YEAR'] = sales_per_month['MONTH'] + ' ' + sales_per_month['YEAR'].astype(str)
    sales_per_month = sales_per_month.drop(['MONTH', 'YEAR'], axis= 1)
    fig2 = px.pie(sales_per_month, names = 'MONTH_YEAR', values = 'AMOUNT', title = "OVERALL SALES DISTRIBUTION" )
    graph2 = fig2.to_html(full_html = False, include_plotlyjs = 'cdn')
    fig3 = px.bar(sales_per_month, x = 'MONTH_YEAR', y = 'AMOUNT', title = "SALES MONTH-BY-MONTH" )
    graph3 = fig3.to_html(full_html = False, include_plotlyjs = 'cdn')


    return render_template('sales.html', 
    table1 = [top5salesproducts.to_html()], 
    table2 = [bottom5salesproducts.to_html()],
    table3 = [sales_per_month.to_html()],  
    table4= [highest_sale_date.to_html()], 
    table5 = [lowest_sale_date.to_html()],
    graph1 = graph1,
    graph2 = graph2,
    graph3 = graph3
    )

@app.route('/features/sales/productid')
def getproductids():
    dataset = pd.read_csv('dataset.csv')
    dataset = dataset.dropna()    
    products_ids = sorted(list(dataset['PRODUCT ID'].unique()))
    return render_template('get_product_ids.html',products_ids = products_ids)   

@app.route('/features/sales/productid/productsales', methods = ['POST'])
def productsales():
    dataset = pd.read_csv('dataset.csv')
    productID= request.form["productids"]
    productID = int(productID)
    product_dataset = dataset[dataset['PRODUCT ID'] == productID]
    product_dataset['DATE'] = pd.to_datetime(product_dataset['DATE'], dayfirst =True)
    # Selected product Name
    product_name = product_dataset['PRODUCT NAME'].unique()[0]
    # Product Price
    product_price = product_dataset['MRP'].unique()[0]
    # Sales per day for selected product
    sales_per_day_product = product_dataset[['DATE', 'AMOUNT']].groupby(by= ['DATE']).sum().reset_index()
    fig1 = px.line(sales_per_day_product, x = 'DATE', y = 'AMOUNT', title = "SALES PER DAY" )
    graph1 = fig1.to_html(full_html = False, include_plotlyjs = 'cdn')
    # Highest Sale Date
    highest_sale_date_product = sales_per_day_product.sort_values(by= ['AMOUNT'], ascending = False).reset_index().drop('index', axis =1)
    highest_sale_date_product = highest_sale_date_product[highest_sale_date_product['AMOUNT'] == highest_sale_date_product['AMOUNT'].max()]
    # Lowest Sale Date
    lowest_sale_date_product = sales_per_day_product.sort_values(by= ['AMOUNT'], ascending = True).reset_index().drop('index', axis =1).head(1)
    lowest_sale_date_product = lowest_sale_date_product[lowest_sale_date_product['AMOUNT'] == lowest_sale_date_product['AMOUNT'].min()]
    # Average Sales per day
    avg_sales_per_day_product = sales_per_day_product['AMOUNT'].mean()


    return render_template('product_sales.html',
    product_name = product_name,
    product_price = product_price,
    table1 = [highest_sale_date_product.to_html()],
    table2 = [lowest_sale_date_product.to_html()],
    avg_sales_per_day_product = avg_sales_per_day_product,
    graph1 = graph1
    )


##############################################################################################
##############################################################################################
# Customer Churn and Segmentation
@app.route('/features/customer-segmentation')
def segmentation():
    # RFM Implementation
    dataset = pd.read_csv('dataset.csv')
    data = dataset
    amount = pd.DataFrame(data.QTY * data.MRP, columns = ['MONETARY'])
    data_cust = np.array(data['CUSTOMER REFERENCE'], dtype=np.object)
    data_cust = pd.DataFrame(data_cust, columns = ["CUSTOMER REFERENCE"])
    data_cust = pd.concat(objs = [data_cust, amount], axis = 1, ignore_index = False)
    monetary = data_cust.groupby(by = ["CUSTOMER REFERENCE"]).MONETARY.sum()
    monetary = monetary.reset_index()
    monetary = monetary[monetary['CUSTOMER REFERENCE'] != 99999]
    frequency = data[['CUSTOMER REFERENCE', 'VOUCHERNO']]
    frequency_df = frequency.groupby("CUSTOMER REFERENCE").VOUCHERNO.count()
    frequency_df = pd.DataFrame(frequency_df)
    frequency_df = frequency_df.reset_index()
    frequency_df.columns = ["CUSTOMER REFERENCE", "FREQUENCY"]
    data["CUSTOMER REFERENCE"] = data["CUSTOMER REFERENCE"].astype(int) 
    data["DATE"] = pd.to_datetime(data["DATE"])
    today_date = dt.datetime(data["DATE"].max().year,data["DATE"].max().month,data["DATE"].max().day) 
    recency = (today_date - data.groupby("CUSTOMER REFERENCE").agg({"DATE":"max"}))
    recency.rename(columns = {"DATE":"RECENCY"}, inplace = True)
    recency_df = recency["RECENCY"].apply(lambda x: x.days)
    RFMScores = frequency_df.merge(monetary, on = "CUSTOMER REFERENCE")
    RFMScores = RFMScores.merge(recency_df, on = "CUSTOMER REFERENCE")
    quantiles = RFMScores.quantile(q=[0.2,0.4,0.6, 0.8])
    quantiles = quantiles.to_dict()
    RFMScores['R'] = RFMScores['RECENCY'].apply(RScore, args=('RECENCY',quantiles,))
    RFMScores['F'] = RFMScores['FREQUENCY'].apply(FScore, args=('FREQUENCY',quantiles,))
    RFMScores['M'] = RFMScores['MONETARY'].apply(MScore, args=('MONETARY',quantiles,))
    RFMScores['RFM Score'] = RFMScores[['R', 'F', 'M']].sum(axis = 1)
    Loyalty_Level = ['Platinum', 'Gold', 'Silver', 'Bronze', 'Iron']
    Score_cuts = pd.qcut(RFMScores['RFM Score'], q = 5, labels = Loyalty_Level)
    RFMScores['RFM Loyalty Level'] = Score_cuts.values
    loyaltylevelswithIDs = RFMScores[['CUSTOMER REFERENCE', 'RFM Loyalty Level']]
    loyaltylevelswithIDs['Number'] = 1
    fig1 = px.pie(loyaltylevelswithIDs, names = 'RFM Loyalty Level', values = 'Number', title = "Loyalty Level Distribution" )
    graph1 = fig1.to_html(full_html = False, include_plotlyjs = 'cdn')
    # Kmeans RFM Clusters
    RFM = RFMScores[['RECENCY', 'FREQUENCY', 'MONETARY']]
    model = KMeans(n_clusters = 5, init= 'k-means++')
    model = model.fit(RFM)
    RFM_kmeans = RFM.copy()
    RFM_kmeans['Cluster ID'] = model.labels_
    RFM_kmeans['RFM Loyalty Level'] = list(RFMScores['RFM Loyalty Level'])
    fig2 = px.scatter_3d(RFM_kmeans, x='FREQUENCY', y='MONETARY', z='RECENCY',color= 'RFM Loyalty Level', title = "Loyalty Level Clusters" )
    graph2 = fig2.to_html(full_html = False, include_plotlyjs = 'cdn')
    # List of customers Loyalty level wise
    platinum = list(loyaltylevelswithIDs[loyaltylevelswithIDs['RFM Loyalty Level'] == 'Platinum']['CUSTOMER REFERENCE'])
    gold = list(loyaltylevelswithIDs[loyaltylevelswithIDs['RFM Loyalty Level'] == 'Gold']['CUSTOMER REFERENCE'])
    silver = list(loyaltylevelswithIDs[loyaltylevelswithIDs['RFM Loyalty Level'] == 'Silver']['CUSTOMER REFERENCE'])
    bronze = list(loyaltylevelswithIDs[loyaltylevelswithIDs['RFM Loyalty Level'] == 'Bronze']['CUSTOMER REFERENCE'])
    iron = list(loyaltylevelswithIDs[loyaltylevelswithIDs['RFM Loyalty Level'] == 'Iron']['CUSTOMER REFERENCE'])
    # Total customers
    total_customers = len(data['VOUCHERNO'].unique())
    # Saving file
    loyaltylevelswithIDs[['CUSTOMER REFERENCE', 'RFM Loyalty Level']].to_csv('Loyalty_levels.csv', encoding = 'utf-8-sig', index = False)
    customer_ids = sorted(list(dataset['CUSTOMER REFERENCE'].unique()))
    
    return render_template('segmentation.html', 
    total_customers = total_customers, graph1 = graph1, graph2 = graph2,
    platinum = platinum, gold = gold,  silver = silver, bronze = bronze, iron = iron,
    customer_ids = customer_ids
     )

@app.route('/features/customer-segmentation/download-loyaltylevel-dataset')
def download_loyalty_level_dataset():
    return send_file('Loyalty_levels.csv',as_attachment = True)

@app.route('/features/customer-segmentation/customized-offers', methods = ['POST'])
def customized_offers():
    dataset = pd.read_csv('dataset.csv')
    dataset = dataset.dropna()    
    custID = request.form["custids"]
    custID = int(custID)
    products = sorted(list(dataset[dataset['CUSTOMER REFERENCE'] == custID]['PRODUCT NAME'].unique()))
    dataset[['PRODUCT NAME', 'MRP']].drop_duplicates(keep='first')
    return render_template('customized_offers.html', products = products, custID = custID)

# Setting Enviroment Variables 
with open('config.json', 'r') as f:
    params = json.load(f)['param']
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = params['gmail-user']
app.config['MAIL_PASSWORD'] = params['gmail-password']
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

@app.route('/features/customer-segmentation/customized-offers/offer-status', methods = ['POST'])
def offer_status():
    dataset = pd.read_csv('dataset.csv')
    series_of_prices = pd.DataFrame(dataset[['PRODUCT NAME', 'MRP']].drop_duplicates(keep='first').reset_index().drop('index', axis =1).groupby('PRODUCT NAME').mean(), columns = ['MRP'])['MRP']
    discount= request.form["discount"]
    discount = int(discount)
    custID = request.form["custid"]
    custID = int(custID)
    level = None
    products = request.form.getlist('checkbox')
    MRP = [series_of_prices[i] for i in products]
    range1 = [i for i in range(len(products))]
    # sending email
    msg = Message("Offer from XYZ store", sender = "custyanalytics@gmail.com", recipients=["custyanalytics@gmail.com"])
    string = "We are pleased to provide you offers on the following products:\n"
    string1 = "Offer: {}".format(discount) + '% Discount\n'
    string2 = ''
    for i in range1:
        string2 = string2 + ' ' + str(products[i]) + '\t\t' + 'MRP:' +str(MRP[i])+ 'Rs.' + '\t'+ 'Discounted Price:' +str(MRP[i] - (discount*0.01*MRP[i]))+ 'Rs.' + '\n'

    msg.body = string + string1 + string2
    mail.send(msg)
    return render_template('offer_status.html', range1 = range1, level=  level, products= products, custID = custID, discount = discount, MRP = MRP)

@app.route('/features/customer-segmentation/<level>')
def loyalty_level(level):
    ll_dataset = pd.read_csv('Loyalty_levels.csv') 
    dataset = pd.read_csv('Dataset.csv') 
    level = str(level)
    custID = sorted(list(ll_dataset[ll_dataset['CUSTOMER REFERENCE'] == level]['CUSTOMER REFERENCE'].unique()))
    products = sorted(list(dataset['PRODUCT NAME'].unique()))

    return render_template('loyalty_level_offers.html', level = level, custID = custID, products = products)

@app.route('/features/customer-segmentation/<level>/offer-status', methods = ['POST'])
def send_email_loyalty_level(level):
    dataset = pd.read_csv('dataset.csv')
    series_of_prices = pd.DataFrame(dataset[['PRODUCT NAME', 'MRP']].drop_duplicates(keep='first').reset_index().drop('index', axis =1).groupby('PRODUCT NAME').mean(), columns = ['MRP'])['MRP']
    discount= request.form["discount"]
    discount = int(discount)
    products = request.form.getlist('checkbox')
    custID = 0
    level = str(level)
    MRP = [series_of_prices[i] for i in products]
    range1 = [i for i in range(len(products))]
    # sending email
    msg = Message("Offer from XYZ store", sender = "custyanalytics@gmail.com", recipients=["custyanalytics@gmail.com"])
    string = "We are pleased to provide you offers on the following products:\n"
    string1 = "Offer: {}".format(discount) + '% Discount\n'
    string2 = ''
    for i in range1:
        string2 = string2 + ' ' + str(products[i]) + '\t\t' + 'MRP:' +str(MRP[i])+ 'Rs.' + '\t'+ 'Discounted Price:' +str(MRP[i] - (discount*0.01*MRP[i]))+ 'Rs.' + '\n'
    msg.body = string + string1 + string2
    mail.send(msg)
    return render_template('offer_status.html', custID = custID, level = level, MRP = MRP, range1 = range1, products= products, discount = discount)
##############################################################################################
##############################################################################################


# Invoice management

app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///invoice.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Invoice(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    current_date = db.Column(db.DateTime, default=datetime.utcnow)
    voucherno = db.Column(db.Integer, nullable=False)
    customerreference = db.Column(db.Integer, nullable=False)
    productid = db.Column(db.Integer, nullable=False)
    productname = db.Column(db.String(200), nullable=False)
    qty = db.Column(db.Integer, nullable=False)
    mrp = db.Column(db.Integer, nullable=False)
    amount = db.Column(db.Integer, nullable=False)

    def __repr__(self) -> str:
        return f"{self.sno}|{self.current_date}|{self.voucherno}|{self.customerreference}|{self.productid}|{self.productname}|{self.qty}|{self.mrp}|{self.amount}"


@app.route('/features/invoicemanagement', methods=['GET', 'POST'])
def invoicemanagement():
    if request.method=='POST':
        voucherno = request.form['voucherno']
        customerreference = request.form['customerreference']
        productid = request.form['productid']
        productname = request.form['productname']
        qty = request.form['qty']
        mrp = request.form['mrp']
        amount = int(qty) * int(mrp)
        invoice_instance = Invoice(
                    voucherno = voucherno,
                    customerreference = customerreference,
                    productid = productid,
                    productname = productname,
                    qty = qty,
                    mrp = mrp,
                    amount = amount
                    )
        db.session.add(invoice_instance)
        db.session.commit()

    allInvoices = Invoice.query.all() 
    # Search Invoice
    dataset = pd.read_csv('dataset.csv')
    dataset = dataset.dropna()    
    unique_invoices = sorted(list(dataset['VOUCHERNO'].unique()))
    #------------------------------------------------------------------------------------
    return render_template('invoice_management.html', allInvoices=allInvoices, unique_invoices = unique_invoices)

@app.route('/features/invoicemanagement/download_database')
def download_database():
    cnx = create_engine('sqlite:///invoice.db').connect()
    df = pd.read_sql_table('invoice', cnx)
    df["current_date"] = pd.to_datetime(df["current_date"])
    df["current_date"] = df["current_date"].dt.date
    df.drop('sno',axis =1, inplace =True)
    df.rename(columns = {
        'current_date' : 'DATE',
        'voucherno' : 'VOUCHERNO',
        'customerreference' : 'CUSTOMER REFERENCE',
        'productid' : 'PRODUCT ID',
        'productname' : 'PRODUCT NAME',
        'qty' : 'QTY',
        'mrp' : 'MRP',
        'amount' : 'AMOUNT'
    }, inplace= True)
    df.to_csv('Database.csv', encoding = 'utf-8-sig', index = False)
    return send_file('Database.csv', as_attachment=True) 

@app.route('/features/invoicemanagement/update/<int:sno>', methods=['GET', 'POST'])
def update(sno):
    if request.method=='POST':
        voucherno = request.form['voucherno']
        customerreference = request.form['customerreference']
        productid = request.form['productid']
        productname = request.form['productname']
        qty = request.form['qty']
        mrp = request.form['mrp']
        amount = int(qty) * int(mrp)
        record = Invoice.query.filter_by(sno=sno).first()
        record.voucherno = voucherno
        record.customerreference = customerreference
        record.productid = productid
        record.productname = productname
        record.qty = qty
        record.mrp = mrp
        record.amount =amount 
        db.session.add(record)
        db.session.commit()
        return redirect("/features/invoicemanagement")
        
    record = Invoice.query.filter_by(sno=sno).first()
    return render_template('update.html', record = record)

@app.route('/features/invoicemanagement/delete/<int:sno>')
def delete(sno):
    record = Invoice.query.filter_by(sno=sno).first()
    db.session.delete(record)
    db.session.commit()
    return redirect("/features/invoicemanagement")

@app.route('/features/invoicemanagement/searchinvoice',  methods=['GET', 'POST'])
def searchinvoice():
    if request.method == 'POST':
        entered_invoice= request.form["unique_invoices"]
        entered_invoice = int(entered_invoice)


        dataset = pd.read_csv('dataset.csv')
        invoice_dataset = dataset[dataset['VOUCHERNO'] ==  entered_invoice]
        invoice_dataset = invoice_dataset.reset_index().drop(['index'], axis = 1)
        date = invoice_dataset['DATE'].unique()[0]
        invoice_number = invoice_dataset['VOUCHERNO'].unique()[0]
        amount = invoice_dataset['AMOUNT'].sum()

        l = [date, invoice_number, amount ]
    return render_template('invoice_search.html', l = l, table1 = [invoice_dataset.to_html()],  title1 = ['first'], title2= ['second'] )

###########################################################################################

if __name__ == '__main__':
    app.run(debug = True)
