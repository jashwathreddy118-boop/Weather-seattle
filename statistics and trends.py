"""
Statistics and trends: four years of daily weather in Seattle, 2012-2015
(Kaggle "Weather Prediction" / seattle-weather.csv, 1,461 days).

Produces three figures for the report:
    categorical_plot.png  - histogram of daily precipitation (dry days shown separately)
    relational_plot.png   - line graph of monthly mean daily max / min temperature
    statistical_plot.png  - heatmap of monthly total precipitation (year x month)

Usage:
    python statistics_and_trends.py [path/to/seattle-weather.csv]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, GREY = '#2a78d6', '#eb6834', '#a8a79f'
SEQ_BLUE = LinearSegmentedColormap.from_list('seq_blue', [
    '#f3f8fe', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf',
    '#184f95', '#0d366b'])
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def set_style():
    """Shared, quiet plotting style sized for a two-page A4 report."""
    plt.rcParams.update({
        'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True,
        'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.dpi': 300, 'figure.dpi': 120,
    })


def plot_categorical_plot(df):
    """Histogram of daily precipitation; dry days drawn as their own bar."""
    fig, ax = plt.subplots(figsize=(6.85, 2.4))
    dry = (df['precipitation'] == 0).sum()
    wet = df.loc[df['precipitation'] > 0, 'precipitation']
    ax.bar(-1.5, dry, width=2.4, color=GREY, edgecolor='white',
           label=f'Dry days, 0 mm (n={dry})')
    ax.hist(wet, bins=np.arange(0, 58, 2), color=BLUE, edgecolor='white',
            linewidth=0.8, label=f'Wet days, 2 mm bins (n={len(wet)})')
    mean_all = df['precipitation'].mean()
    ax.axvline(mean_all, color=INK, linestyle=':', linewidth=1.2)
    ax.text(mean_all + 0.6, 600, f'mean of all days\n{mean_all:.2f} mm',
            fontsize=7, va='top')
    ax.set_yscale('log')
    ax.set_ylim(0.8, 1400)
    ax.set_xlim(-3.2, 58)
    ax.set_xlabel('Daily precipitation (mm)')
    ax.set_ylabel('Number of days (log scale)')
    ax.set_title('Distribution of daily precipitation, 2012-2015', loc='left')
    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig('categorical_plot.png')
    plt.close(fig)


def plot_relational_plot(df):
    """Line graph of the monthly means of daily max and min temperature."""
    monthly = df.set_index('date')[['temp_max', 'temp_min']].resample('MS').mean()
    rolling = df.set_index('date')['temp_max'].rolling('365D').mean()
    rolling = rolling[rolling.index >= '2013-01-01']
    fig, ax = plt.subplots(figsize=(6.85, 2.6))
    ax.fill_between(monthly.index, monthly['temp_min'], monthly['temp_max'],
                    color=GRID, alpha=0.6, linewidth=0)
    ax.plot(monthly.index, monthly['temp_max'], color=ORANGE, linewidth=1.8,
            marker='o', markersize=3, label='Mean daily maximum')
    ax.plot(monthly.index, monthly['temp_min'], color=BLUE, linewidth=1.8,
            marker='s', markersize=3, label='Mean daily minimum')
    ax.plot(rolling.index, rolling.values, color=INK2, linewidth=1.2,
            linestyle='--', label='365-day rolling mean of daily max')
    for year, value in df.groupby('year')['temp_max'].mean().items():
        ax.text(pd.Timestamp(f'{year}-07-01'), 30.4,
                f'{year} mean max {value:.1f}°C',
                ha='center', fontsize=6.8, color=INK2)
    ax.set_ylim(-1, 33)
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Monthly mean of daily maximum and minimum temperature',
                 loc='left')
    ax.legend(frameon=False, loc='upper center', ncol=3,
              bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig('relational_plot.png')
    plt.close(fig)


def plot_statistical_plot(df):
    """Heatmap of total precipitation in each month of each year."""
    table = df.pivot_table(index='year', columns='month', values='precipitation',
                           aggfunc='sum')
    table.columns = MONTHS
    fig, ax = plt.subplots(figsize=(6.85, 1.95))
    sns.heatmap(table, annot=True, fmt='.0f', cmap=SEQ_BLUE, vmin=0, vmax=290,
                linewidths=1.5, linecolor='white', annot_kws={'fontsize': 7},
                cbar_kws={'label': 'Total (mm)', 'pad': 0.015}, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='both', length=0)
    ax.tick_params(axis='y', rotation=0)
    ax.grid(False)
    ax.set_title('Monthly total precipitation (mm)', loc='left')
    fig.tight_layout()
    fig.savefig('statistical_plot.png')
    plt.close(fig)


def statistical_analysis(df, col):
    """Return the first four moments of a column (excess kurtosis)."""
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Parse dates, add year/month, and check the weather labels."""
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset='date').sort_values('date')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    print(df.describe().round(2))
    print(df[['precipitation', 'temp_max', 'temp_min', 'wind']].corr().round(3))
    print(pd.crosstab(df['year'], df['weather']))
    print(pd.crosstab(df['weather'], df['precipitation'] > 0))
    return df


def writing(moments, col):
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    skew = 'right' if moments[2] > 0.5 else 'left' if moments[2] < -0.5 else 'not'
    kurt = ('leptokurtic' if moments[3] > 1 else
            'platykurtic' if moments[3] < -1 else 'mesokurtic')
    print(f'The data was {skew} skewed and {kurt}.')


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'seattle-weather.csv'
    set_style()
    df = preprocessing(pd.read_csv(path))
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    writing(statistical_analysis(df, 'precipitation'), 'precipitation')
    wet = df[df['precipitation'] > 0]
    writing(statistical_analysis(wet, 'precipitation'), 'precipitation (wet days)')
    writing(statistical_analysis(df, 'temp_max'), 'temp_max')
    print('dry-day share:', round((df['precipitation'] == 0).mean(), 3))
    print(df.groupby('year')['precipitation'].sum().round(1))
    print(df.groupby('month')[['temp_max', 'temp_min', 'precipitation']].mean().round(2))


if __name__ == '__main__':
    main()
