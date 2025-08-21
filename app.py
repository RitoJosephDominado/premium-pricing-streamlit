import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

import matplotlib.ticker as ticker
import statsmodels.api as sm
import statsmodels.formula.api as smf

import plotly.express as px


from streamlit_extras.stylable_container import stylable_container
# from streamlit.extras import e
from eda import create_missing_values_barplot
from eda import create_damage_histogram
from eda import create_age_plotly
from eda import create_damage_below_15000_histogram
from eda import create_age_histogram
from eda import plot_categorical_barplots
from eda import create_age_plotly
from eda import create_damage_plotly

from modeling import prepare_data
from modeling import create_premium_df
from modeling import print_error_metrics
from modeling import create_glm_model

box_css = """
    {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 0.5rem;
        padding: calc(1em - 1px);
        background-color: rgba(170, 255, 170, 0.8);
    }
    """


green_box_css="""
{
  background-color: #e0f2e0; /* pastel orange */
  padding: 2.5rem;
  border-radius: 1.5rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1),
              0 1px 3px rgba(0, 0, 0, 0.08);
  border: 1px solid rgba(0, 0, 0, 0.05);
  margin: 2rem auto;
  max-width: 900px;
}
"""

box_css = """
{
  /* Set the background color to a light pastel green */
  background-color: #e0f2e0; /* A soft, calming green */
  
  /* Add some internal padding for spacing */
  padding: 2.5rem; /* Large padding for a clean, spacious feel */
  
  /* Give it rounded corners */
  border-radius: 1.5rem; /* Generous rounding for a soft look */
  
  /* Use a box shadow for a 'floating' effect */
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), /* Main shadow */
              0 1px 3px rgba(0, 0, 0, 0.08); /* Secondary, lighter shadow */
  
  /* Add a very subtle border */
  border: 1px solid rgba(0, 0, 0, 0.05); /* A thin, barely-there border */
  
  /* Center the box and give it a max width for responsiveness */
  margin: 2rem auto;
  max-width: 900px;
}
"""

orange_box_css="""
{
  background-color: #f2e0d3; /* pastel orange */
  padding: 2.5rem;
  border-radius: 1.5rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1),
              0 1px 3px rgba(0, 0, 0, 0.08);
  border: 1px solid rgba(0, 0, 0, 0.05);
  margin: 2rem auto;
  max-width: 900px;
}
"""

st.set_page_config(layout="wide")
df = pd.read_csv('AutoBI_output.csv')

plotting_df = df.copy()
categorical_cols = ['VEHICLE_TYPE', 'GENDER', 'MARITAL_STATUS', 'PREVCLM', 'SEATBELT']
for col in categorical_cols:
    plotting_df[col] = plotting_df[col].fillna('Missing')

feature_df = df.loc[:, ['INSAGE', 'VEHICLE_TYPE', 'GENDER', 'MARITAL_STATUS', 'PREVCLM', 'SEATBELT']]
target_series = df.loc[:, 'LOSS']


# def create_glm_model(target_series, prepared_feature_df, model_type='OLS'):
#     if model_type == 'OLS':
#         model = sm.OLS(target_series, prepared_feature_df).fit()
#     elif model_type == 'Gamma':
#         model = sm.GLM(target_series, prepared_feature_df, family=sm.families.Gamma(link=sm.families.links.Log())).fit()
#     elif model_type == 'Tweedie':
#         model = sm.GLM(target_series, prepared_feature_df, family=sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log())).fit()
#     return model


data_tab, eda_tab, model_tab, final_premium_tab = st.tabs(['Data', 'EDA', 'OLS Model', 'Final Premium'])

with data_tab:
    st.header('Technical Exam by Rito Dominado')
    st.image('car_crash_vecteezy.jpg',width=300)
    with stylable_container(
            key="container_with_border",
            css_styles = green_box_css
        ):
        
        st.dataframe(df)

with eda_tab:
    st.header('Exploratory Data Analysis 📊')
    with stylable_container(key='eda_container', css_styles=box_css):
        st.subheader('Missing Values')
        y_vals = np.arange(0, 0.16, 0.01)
        missing_data_df = df.loc[:, ['ACTUALDAMAGE', 'INSAGE','VEHICLE_TYPE', 'GENDER', 'MARITAL_STATUS','PREVCLM', 'SEATBELT']].isnull().sum()/df.shape[0]
        sns.barplot(missing_data_df)
        plt.title('')
        plt.xlabel('Percentage of Missing Values')
        plt.ylabel('Columns')
        plt.yticks(y_vals, labels=[f'{int(x*100)}%' for x in y_vals])

        st.write('This bar plot shows the percentage of missing values in each column.')
        fig = create_missing_values_barplot(df)
        

    numeric_eda_cols = st.columns([0.5, 0.5])
    categorical_eda_1_cols = st.columns([0.333, 0.333, 0.333])
    categorical_eda_2_cols = st.columns([0.6, 0.4])
    with numeric_eda_cols[0]:
        with stylable_container(key='damage_container', css_styles=box_css):
            st.subheader('Damage (ACTUALDAMAGE)')
            st.write('')

            fig = px.histogram(df, x='ACTUALDAMAGE', nbins=20, title='Distribution of Damages')
            fig.update_traces(marker_line_width=1, marker_line_color="black")
            fig.update_layout(xaxis_title='Damage', yaxis_title='Count')
            st.plotly_chart(fig, key='damage_plotly')

    with numeric_eda_cols[1]:
        with stylable_container(key='age_container', css_styles=box_css):
            st.subheader('Age (INSAGE)')
            st.write('')
            st.html('''<ul>
            <li>This is the only numerical predictor in the dataset.
            <li>This has 14% of its values missing, the most of any column.
            </ul>''')
            fig = px.histogram(df, x='INSAGE', nbins=20, title='Distribution of Ages')
            fig.update_traces(marker_line_width=1, marker_line_color="black")
            fig.update_layout(xaxis_title='Age', yaxis_title='Count')
            st.plotly_chart(fig, key='age_plotly')
    
    with categorical_eda_1_cols[0]:
        fig = px.bar(plotting_df, x='VEHICLE_TYPE', title='Vehicle Types', color='VEHICLE_TYPE',color_discrete_sequence=px.colors.qualitative.Pastel)
        with stylable_container(key='vehicle_type_container', css_styles=orange_box_css):
            st.subheader('Vehicle Type')
            st.html('''<ul>
            <li>Customers with small cars ended up with higher damages than those with big cars.
            </ul>''')
            st.plotly_chart(fig, key='vehicle_type_plotly')
        
    with categorical_eda_1_cols[1]:
        with stylable_container(key='gender_container', css_styles=orange_box_css):
            st.subheader('Gender')
            st.html('''<ul>
            <li>There are slightly more female customers than male ones.
            </ul>''')
            fig = px.bar(plotting_df, x='GENDER', title='Genders', color='GENDER',color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig,key='gender_plotly')

    with categorical_eda_1_cols[2]:
        with stylable_container(key='prevclm_container', css_styles=orange_box_css):
            st.subheader('Previous Claims (PREVCLM)')
            st.html('''<ul>
            <li>The most of the customers are first-time claimants.
            </ul>''')
            fig = px.bar(plotting_df, x='PREVCLM', title='Previous Claims Options',color='PREVCLM',color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, key='prevclm_plotly')

    with categorical_eda_2_cols[0]:
        with stylable_container(key='marital_status_container', css_styles=box_css):
            st.subheader('Marital Status')
            st.write(' ')
            st.html('''<ul>
            <li>Married and single customers were the largest groups, while divorced/separated and widowed customers are a minority.
            </ul>''')
            fig = px.bar(plotting_df, x='MARITAL_STATUS', title='Marital Statuses', color='MARITAL_STATUS', color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, key='marital_status_plotly')

    with categorical_eda_2_cols[1]:
        with stylable_container(key='seatbelt_container', css_styles=box_css):
            st.subheader('Seatbelt Usage')
            st.html('''<ul>
            <li>The vast majority of customers used a seatbelts during the accident.
            </ul>''')
            
            fig = px.bar(plotting_df, x='SEATBELT', title='Seatbelt Options',color='SEATBELT',color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, key='seatbelt_plotly')


with model_tab:
    # select imputation method for categorical and numerical variables
    # 
    # select model type
    # features selected
    # % test data
    st.header('Modeling 📈')
    # st.sidebar()
    with stylable_container(key='options_container', css_styles=green_box_css):
        
        st.subheader('Modeling Options')
        modeling_options_cols = st.columns([0.5, 0.2, 0.4])
        with modeling_options_cols[0]:
            st.write('Select the options for the model:')
            perc_testing = st.slider('Percentage of data to be used for testing:', min_value=10, max_value=90, value=30, step=5, key='test_size_slider',format="%.1f%%")
            # imputation_method = st.selectbox('Select Imputation Method', options=['KNN', 'Simple'], key='imputation_method')
            # age_imputation_method = st.selectbox('Select Age Imputation Method', options=['Mean', 'Median'], key='age_imputation_method')
            model_name = st.selectbox('Select Model Type', options=['OLS', 'Gamma', 'Tweedie'], key='model_type')
        with modeling_options_cols[2]:
            st.text('Predictors to include:')
            include_age = st.checkbox('Age (INSAGE)', value=True, key='include_age_checkbox')
            include_vehicle_type = st.checkbox('Vehicle Type (VEHICLE_TYPE)', value=True, key='include_vehicle_type_checkbox')
            include_gender = st.checkbox('Gender (GENDER)', value=True)
            include_marital_status = st.checkbox('Marital Status (MARITAL_STATUS)', value=True)
            include_seatbelt = st.checkbox('Seatbelt Usage (SEATBELT)', value=True)
            include_prevclm = st.checkbox('Had Previous Claims (PREVCLM)', value=True)
        st.button('Run Model', key='run_model_button')

    with stylable_container(key='data_split_container', css_styles=green_box_css):
        st.subheader('Selected Settings')
        st.subheader('Testing Set: ' + str(perc_testing) + '%')
        st.write(f'Model Selected: {model_name}')

    with stylable_container(key='data_preparation_container', css_styles=green_box_css):
        st.subheader('Data Preparation')
        st.write('This section prepares the data for modeling by imputing missing values and encoding categorical variables.')
        st.write('The prepared data is then used to create a Generalized Linear Model (GLM).')
        st.write('The GLM can be used to predict the expected loss based on the features.')
        
    feature_train_df, feature_test_df, target_train_series, target_test_series = train_test_split(
        feature_df, target_series, test_size=perc_testing/100, random_state=12345)
    
    predictors_to_remove = []
    if not include_age:
        predictors_to_remove.append('INSAGE')
    if not include_vehicle_type:
        predictors_to_remove.append('VEHICLE_TYPE')
    if not  include_marital_status:
        predictors_to_remove.append('MARITAL_STATUS') 
    if not include_gender:
        predictors_to_remove.append('GENDER')
    if not include_seatbelt:
        predictors_to_remove.append('SEATBELT')
    if not include_prevclm:
        predictors_to_remove.append('PREVCLM')

    prepared_feature_train_df = prepare_data(feature_train_df)
    prepared_feature_test_df = prepare_data(feature_test_df)
        
    model = create_glm_model(target_train_series, prepared_feature_train_df, model_type=model_name)
    train_pred_series = model.predict(prepared_feature_train_df)

    train_col, test_col = st.columns([0.5, 0.5])
    with train_col:
        with stylable_container(key='train_data_container', css_styles=green_box_css):
            st.subheader('Training Data')
            st.dataframe(prepared_feature_train_df)
            
    with test_col:
        with stylable_container(key='test_data_container', css_styles=orange_box_css):
            st.subheader('Test Data')
            st.dataframe(prepared_feature_test_df)
            

            # st.write('Model Predictions')
            # st.write(model.predict(prepared_feature_test_df))
            # st.write('Model Error Metrics')
        # print_error_metrics(test_target_series, model.predict(prepared_feature_test_df))

    with st.expander('Model Summary (straight from model.summary() function in statsmodels)'):
        # with stylable_container(key='model_summary_container', css_styles=orange_box_css):
        #     st.subheader('Model Summary')
        st.write(model.summary())


with final_premium_tab:
    st.header('Final Premium Calculation')
    st.subheader('Note that the values here depend on the model used in the Modeling tab.')
    premium_df = create_premium_df(prepared_feature_test_df, target_test_series, model.predict(prepared_feature_test_df))
    st.dataframe(premium_df)


