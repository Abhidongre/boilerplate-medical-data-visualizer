import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 - Import data
df = pd.read_csv('medical_examination.csv')

# 2 - Add 'overweight' column
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)).apply(lambda x: 1 if x > 25 else 0)

# 3 - Normalize data
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4 - Categorical Plot Function
def draw_cat_plot():
    # 5 - Create DataFrame for cat plot
    df_cat = pd.melt(df, 
                     id_vars=['cardio'], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6 - Group and reformat data
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7 - Draw the catplot
    plot = sns.catplot(
        data=df_cat, 
        x='variable', 
        y='total', 
        hue='value', 
        col='cardio', 
        kind='bar'
    )

    # 8 - Get figure
    fig = plot.fig

    # 9 - Save figure
    fig.savefig('catplot.png')
    return fig

# 10 - Heat Map Function
def draw_heat_map():
    # 11 - Clean data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12 - Calculate correlation matrix
    corr = df_heat.corr()

    # 13 - Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14 - Set up figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15 - Draw heatmap
    sns.heatmap(corr, 
                mask=mask, 
                annot=True, 
                fmt=".1f", 
                square=True, 
                linewidths=0.5, 
                cbar_kws={"shrink": 0.5}, 
                center=0,
                ax=ax)

    # 16 - Save figure
    fig.savefig('heatmap.png')
    return fig
