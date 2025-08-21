import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
from statsmodels.iolib.smpickle import save_pickle
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

def prepare_data(df):
    categorical_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    numerical_imputer = SimpleImputer(missing_values=np.nan, strategy='median')

    categorical_df = df.drop('INSAGE', axis=1)
    numerical_df = df.loc[:, ['INSAGE']]

    imputed_categorical_df = pd.DataFrame(categorical_imputer.fit_transform(categorical_df), columns=categorical_df.columns,index=df.index)
    imputed_numerical_df = pd.DataFrame(np.round(numerical_imputer.fit_transform(numerical_df)), columns=numerical_df.columns, index=df.index)

    encoder = OneHotEncoder(sparse_output=False)
    encoded_df = pd.DataFrame(encoder.fit_transform(imputed_categorical_df), columns = encoder.get_feature_names_out(), index=df.index)
    encoded_df = encoded_df.drop(columns=['PREVCLM_No', 'MARITAL_STATUS_Single', 'GENDER_Female', 'SEATBELT_Yes', 'VEHICLE_TYPE_Small car'])
    
    imputed_df = encoded_df
    imputed_df['INSAGE'] = imputed_numerical_df.INSAGE
    ols_df = sm.add_constant(imputed_df)
    return ols_df

def create_premium_df(df, target_series, expected_loss_series):
    df = df.copy().reset_index()
    expected_loss_series.index = df.index
    df['target'] = target_series
    df['expected_loss'] = expected_loss_series
    df['diff'] = expected_loss_series - target_series


    df['commission'] = df.expected_loss*0.2
    df['reinsurance'] = df.expected_loss*0.1
    df['admin'] = df.expected_loss*0.1
    df['profit_margin'] = df.expected_loss*0.05
    df['final_premium'] =  df['expected_loss'] + df['commission'] + df['reinsurance'] + df['admin'] + df['profit_margin']+ + 10
    return df

def print_error_metrics(target_series, pred_series):
    mape = mean_absolute_percentage_error(target_series, pred_series)
    mse = mean_squared_error(target_series, pred_series)
    print(f'Mean Absolute Percent Error: {mape}')
    print(f'Mean Squared Error: {mse}')

def create_glm_model(target_series, prepared_feature_df, model_type='OLS'):
    if model_type == 'OLS':
        model = sm.OLS(target_series, prepared_feature_df).fit()
    elif model_type == 'Gamma':
        model = sm.GLM(target_series, prepared_feature_df, family=sm.families.Gamma(link=sm.families.links.Log())).fit()
    elif model_type == 'Tweedie':
        model = sm.GLM(target_series, prepared_feature_df, family=sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log())).fit()
    return model