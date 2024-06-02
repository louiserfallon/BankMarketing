"""
Helpers for plotting categorical and numerical features
for a classification problem with a binary "success" outcome
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.dates import DateFormatter, MonthLocator


def save_to_figs(plt, col, chart_name, suffix=''):
    """
    makes sure the figures folder exists, and saves a plot there with high quality

    :param plt: the plot to save
    :param col: the column being considered, to contatenate in the filename
    :param chart_name: name of the  to save the chart
    :param suffix: a suffix for a chart name, if needed
    :return: None
    """
    if not os.path.exists('figures'):
        os.makedirs('figures')
    # set a high resolution for zooming in
    plt.savefig(f'./figures/{chart_name}_{col}_{suffix}.png', dpi=600)
    plt.clf()


def export_num_chart(df, col, min_bucket=15, accent_colour='#ff6002'):
    """
    Generates & saves a line chart for a numerical variable on the x-axis
    With the success rate on the y-axis
    And a dual column chart axis showing the # of examples
    Note: this function writes a plot to a file

    :param df: input dataframe
    :param col: column to group by
    :param min_bucket: the minimum number of examples for a line to show on the y-axis
    :param accent_colour: colour to accent
    :return:
    """
    df = df.copy(deep=True)

    # Convert 'yes'/'no' to binary
    df['success'] = df['success'].map({'yes': 1, 'no': 0})
    # Remove outliers for visualization
    p95 = df[col].quantile(0.95)
    df_clean = df[(df[col] >= 0) & (df[col] <= p95)]

    # Aggregate for each value of the column
    grouped = df_clean.groupby(col)['success'].agg(['mean', 'count'])
    # Replace mean values where count is below threshold with NaN
    grouped.loc[grouped['count'] < min_bucket, 'mean'] = np.nan

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot bars for count
    positions = np.arange(len(grouped.index))
    ax1.bar(positions, grouped['count'], color='darkgrey', alpha=0.6, width=0.4)  # Adjust width as needed
    ax1.set_ylabel('Attempts', color='darkgrey')
    ax1.set_xlabel('Days since Last Contact (if contacted)', color='black')
    ax1.tick_params(axis='y', labelcolor='darkgrey')

    # Create a second y-axis to plot the proportion/mean
    ax2 = ax1.twinx()
    ax2.plot(positions, grouped['mean'], color=accent_colour, label='Proportion of Success')  # Example accent color
    ax2.set_ylabel('Proportion of Success', color=accent_colour)
    ax2.tick_params(axis='y', labelcolor=accent_colour)
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    # Set labels and title
    plt.xlabel('Days since last campaign')
    plt.title('Proportion of successful direct marketing attempts by time since last contact')

    # Aesthetics
    plt.grid(False)
    fig.tight_layout()
    for ax in (ax1, ax2):
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_tick_params(size=0)
        ax.yaxis.set_tick_params(size=0)

    # Save to png
    save_to_figs(plt, col, 'numeric')


def export_time_chart(df, col='date', min_bucket=15, accentcolor='#ff6002'):
    """
    Generates & saves a time series chart for a date variable on the x-axis
    With the success rate on the y-axis
    And a dual column chart axis showing the # of examples
    Note: this function writes a plot to a file

    :param df: input dataframe
    :param col:column to group by
    :param min_bucket: the minimum number of examples for a line to show on the y-axis
    :param accentcolor: colour to accent the y-axis and axis title
    :return: None
    """
    df.set_index('date', inplace=True)

    # Resample the data by week and calculate the proportion of 'yes' values and the total counts
    weekly_df = df.resample('W').agg(
        proportion=('success', lambda x: (x == 'yes').mean()),
        total=('success', 'size')
    ).reset_index()

    # 'break' the line chart by showing null
    # when there isn't enough data that time period
    weekly_df.loc[weekly_df['total'] < min_bucket, 'proportion'] = np.nan

    # Initialise chart
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot the total number of observations as a bar chart
    ax1.bar(weekly_df['date'], weekly_df['total'], color='darkgrey', alpha=0.6, width=5)
    ax1.set_ylabel('Attempts', color='darkgrey')
    ax1.tick_params(axis='y', labelcolor='darkgrey')

    # Set locator for x-axis to show one tick per month
    date_form = DateFormatter("%b-%y")
    ax1.xaxis.set_major_formatter(date_form)
    ax1.xaxis.set_major_locator(MonthLocator())
    plt.xticks(rotation=90, fontstyle='italic')

    # Create a second y-axis to plot the proportion
    ax2 = ax1.twinx()
    ax2.plot(weekly_df['date'], weekly_df['proportion'], color=accentcolor, label='Proportion of Success')
    ax2.set_ylabel('Proportion of Success', color=accentcolor)
    ax2.tick_params(axis='y', labelcolor=accentcolor)
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    # Aesthetics
    plt.title('Proportion of successful direct marketing attempts per week')
    fig.tight_layout()
    for ax in (ax1, ax2):
        ax.grid(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_tick_params(size=0)
        ax.yaxis.set_tick_params(size=0)

    # Save to png
    save_to_figs(plt, col, 'ts')


def export_cat_chart(df, col, order_asc=False, cuts=None):
    """
    Generates & saves a time series chart for a date variable on the x-axis
    With the success rate on the y-axis
    And a dual column chart axis showing the # of examples
    Note: this function writes a plot to a file

    :param df: input dataframe
    :param col:column to group by
    :param order_asc: flag for whether to order by the category order
            otherwise orders by the success rate
    :param cuts: optional parameter for a list of bin/cut edges, to categorise a numerical value
    :return: None
    """
    # Protect the original df from being edited
    df = df.copy(deep=True)

    # Change numerical to categorical based on cuts/bins if provided
    if cuts is not None:
        df[col] = pd.cut(df[col], bins=cuts,
                         labels=[f'{cuts[i]}-{cuts[i + 1]}' for i in range(len(cuts) - 1)])

    # Calculate success rate and count
    df['success'] = df['success'].map({'yes': 1, 'no': 0})
    success_rate = df.groupby(col, observed=False)['success'].mean()
    count = df[col].value_counts()

    # Merge success rate and count into a DataFrame
    summary_df = pd.DataFrame({'success_rate': success_rate, 'count': count})
    # Sort by success rate
    if order_asc:
        summary_df = summary_df.sort_index()
    else:
        summary_df = summary_df.sort_values(by='success_rate', ascending=True)

    # Plot the success rate
    ax = summary_df['success_rate'].plot(kind='barh', color='#ff6002', linewidth=0)

    # Annotate the count on top of the bars
    for i, (idx, row) in enumerate(summary_df.iterrows()):
        ax.text(row['success_rate'] + 0.005, i, f'{row['success_rate']:.0%}, n={row["count"]:.0f}', va='center')

    # Set labels and title
    plt.title('Proportion of successful direct marketing attempts per category')
    ax.set_title(f'Success Rate and Count by {col.replace('_', ' ').title()}')
    ax.set_xlabel('Success Rate')
    ax.set_ylabel(col.replace('_', ' ').title())
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    # Aesthetics
    plt.tight_layout()
    ax.grid(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(size=0)
    # Extend the x-axis limit to accommodate annotations
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax * 1.2)

    # Save to png
    save_to_figs(plt, col, 'cat_chart')
    plt.close()
