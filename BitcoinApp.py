import streamlit as st
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime
import seaborn as sns

color = sns.color_palette()
import matplotlib.pyplot as plt


ROOT_URL = "F:/Content/MyLearn/DT/streamlit/Simple_demo/dataset/bitcoin_price.csv"
ROOT_URL_DATASET = "F:/Content/MyLearn/DT/streamlit/Simple_demo/dataset/bitcoin_dataset.csv"


# FILES = ["imdb_labelled.txt", "amazon_cells_labelled.txt", "yelp_labelled.txt"]


# @st.cache(allow_output_mutation=True)
def get_all_data():
    """Loads the source data"""
    data = pd.read_csv(ROOT_URL)
    return data


@st.cache()
def preprocessing_data():
    processed_data = pd.read_csv(ROOT_URL)

    for i, row in processed_data.iterrows():
        dt = datetime.strptime(row['Date'], '%b %d, %Y')
        dt = dt.strftime('%Y-%m-%d')
        row['Date'] = dt
        processed_data.set_value(i, 'Date', dt)

    return processed_data


@st.cache()
def preprocessing_data_second():
    processed_dataset = pd.read_csv(ROOT_URL_DATASET)

    for i, row in processed_dataset.iterrows():
        dt = datetime.strptime(row['Date'], '%Y-%m-%d 00:00:00')
        dt = dt.strftime('%Y-%m-%d')
        row['Date'] = dt
        processed_dataset.set_value(i, 'Date', dt)

    return processed_dataset


@st.cache()
def first_step():
    df_price = preprocessing_data()
    df_dataset = preprocessing_data_second()
    joined_data = df_price.merge(df_dataset, on='Date')
    joined_data['year'] = pd.to_datetime(joined_data['Date']).dt.year
    return joined_data


def convert_to_year(data):
    data_year = pd.to_datetime(data['Date']).dt.year
    data_year.columns=['Year']
    return data_year


st.title('Bitcoin Analysis')

with st.spinner("Extracting source data..."):
    # all_data = get_all_data()
    source_data = first_step()
    print(source_data)
    # source_data["sentiment"] = source_data["sentiment"].map(
    #     {"0": "Negative", "1": "Positive"}
    # )
    #st.info(f"{len(source_data.index)} rows were extract with **success**!")

top = st.selectbox(
    "Select number of rows to show", [5, 10, 25, 50, 100, len(source_data)]
)

st.table(source_data.head(top))

price_type = st.radio('Select price type', ('Open', 'Close', 'High', 'Low'))
# Plot
sns.distplot(source_data[price_type], kde=False, label=price_type+' price')  # default bins using Freedman-Diaconis rule.
# sns.distplot(joined_data['Open'], kde=False, label='Open price') #default bins using Freedman-Diaconis rule.
# sns.distplot(joined_data['High'], kde=False, label='High price') #default bins using Freedman-Diaconis rule.
plt.title('Distribution of '+price_type+ ' price of Bitcoin')
plt.legend(loc='best')
st.pyplot()


all_columns=['Close','btc_market_cap',
                            'btc_avg_block_size',
                            'btc_n_transactions_per_block',
                            'btc_hash_rate',
                            'btc_difficulty',
                            'btc_cost_per_transaction',
                            'btc_n_transactions']

#selected_col = source_data[all_columns]
selected_col = source_data[st.multiselect("Write Column name", all_columns, default=all_columns)]
my_corrmat=selected_col.corr()

#
# #selected_col.head()
# corrmat = selected_col.corr(method='pearson')
#
# columns = ['Close']
# my_corrmat = corrmat.copy()
# mask = my_corrmat.columns.isin(columns)
# my_corrmat.loc[:, ~mask] = 0
# #print(my_corrmat)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(my_corrmat, annot=False, fmt="f", cmap="Blues") #vmax=1., square=True)
plt.title("Correlation Between Price and other factors", fontsize=15)
#plt.savefig('variablecorrelation.png', bbox_inches='tight')
st.pyplot()
