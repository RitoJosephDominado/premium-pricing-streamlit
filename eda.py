import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def create_damage_histogram(df):
    bins= range(0, 280001, 15000)
    plt.figure(figsize=(10, 7))
    damage_hist = sns.histplot(df.query(' POLICYNUM != 22286').ACTUALDAMAGE, bins=bins)
    damage_hist.set_title('Distribution for Damages')
    damage_hist.set_xticks(bins)
    plt.xticks(bins, rotation =45)
    damage_hist.set_xlabel('Damages')
    plt.show()

def create_damage_plotly(df):
    fig = px.histogram(df, x='ACTUALDAMAGE', nbins=20, title='Distribution of Damages')
    fig.update_traces(marker_line_width=1, marker_line_color="black")
    fig.update_layout(xaxis_title='Damage', yaxis_title='Count')
    return fig

def create_damage_below_15000_histogram(df):
    bins= range(0, 1501, 100)
    plt.figure(figsize=(10, 7))
    damage_hist = sns.histplot(df.query('ACTUALDAMAGE <= 1500').ACTUALDAMAGE, bins=bins)
    damage_hist.set_title('Distribution for Damages Below $1,500 (Bottom 92%)')
    damage_hist.set_xticks(bins)
    damage_hist.set_xlabel('Damage')
    plt.show()


def create_age_histogram(df):
    plt.figure(figsize=(10, 7))
    bins = range(0, 100, 5)
    sns.histplot(df.INSAGE,bins=bins)
    plt.xticks(bins)

    plt.title('Distribution of Ages')
    plt.xlabel('Age')
    plt.show()
    plt.show()

def plot_categorical_barplots(df):
    fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize=(15, 12))
    axes = axes.flatten()
    categorical_columns = ['VEHICLE_TYPE', 'GENDER', 'MARITAL_STATUS', 'PREVCLM', 'SEATBELT']
    for i, col in enumerate(categorical_columns):
        plot_df = df[col].value_counts(dropna=False).reset_index()
        plot_df.columns = ['Category', 'Count']
        plot_df['Category'] = plot_df['Category'].fillna('Missing')
        if (col == 'SEATBELT' or col == 'PREVCLM'):
            plot_df['Category'] = pd.Categorical(plot_df['Category'], categories=['Yes', 'No', 'Missing'])
        if (col == 'MARITAL_STATUS'):
            plot_df['Category'] = pd.Categorical(plot_df['Category'], categories=['Single', 'Married', 'Divorced/Separated', 'Widowed', 'Missing'])
        sns.barplot(x='Category',y='Count', data=plot_df, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
    plt.tight_layout()
    if len(categorical_columns) < len(axes):
        axes[-1].remove()
    plt.show()

def create_categorical_violinplots(df):
    num_cols = 3
    num_rows = 2
    categorical_columns = ['VEHICLE_TYPE', 'GENDER', 'MARITAL_STATUS', 'PREVCLM', 'SEATBELT']
    plt.figure(figsize=(18, 8))
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5 * num_cols, 6))
    axes = axes.flatten()
    display_df = df.query('ACTUALDAMAGE <= 200000')
    for i, col in enumerate(categorical_columns):
        means = display_df.groupby(col)['ACTUALDAMAGE'].median()
        for j, mean_val in enumerate(means):
            axes[i].axhline(y=mean_val, linestyle='--', xmin=j/(len(means))+0.1, xmax=(j+1)/(len(means))-0.1, color='black', linewidth=1)
        sns.violinplot(x=col, y='ACTUALDAMAGE', data=display_df, ax=axes[i], palette='pastel')

        axes[i].set_title(f'Distribution of DAMAGE by {col}', fontsize=14)
    plt.tight_layout()
    if len(categorical_columns) < len(axes):
        axes[-1].remove()

def create_missing_values_barplot(df):
    plt.figure(figsize=(10, 7))
    sns.barplot(df.loc[:, ['INSAGE','VEHICLE_TYPE', 'GENDER', 'MARITAL_STATUS','PREVCLM', 'SEATBELT']].isnull().sum()/df.shape[0])
    plt.title('Percentage of Missing Values')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    y_vals = np.arange(0, 0.16, 0.01)
    plt.yticks(y_vals, labels=[f'{int(x*100)}%' for x in y_vals])
    plt.show()

def create_age_plotly(df):
    fig = px.histogram(df, x='INSAGE', nbins=20, title='Distribution of Ages')
    fig.update_traces(marker_line_width=1,marker_line_color="black")
    fig.update_layout(xaxis_title='Age', yaxis_title='Count')
    return fig

