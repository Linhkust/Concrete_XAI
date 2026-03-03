import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from taylorDiagram import TaylorDiagram
from statsmodels.stats.outliers_influence import variance_inflation_factor

def viz(df, cat):
    g = sns.FacetGrid(df, row=cat, height=1.5, aspect=6)
    g.map(sns.kdeplot, "Compressive strength", fill=True)
    plt.show()

def taylor_diagram(df):
    stdref = df.loc[0, 'ref']

    # Samples std,rho,name
    samples = df.loc[:, ['std', 'rho', 'Method']]

    samples = samples.values.tolist()

    fig = plt.figure(figsize=(10, 8))
    dia = TaylorDiagram(stdref, fig=fig, label='Reference', extend=False)
    dia.samplePoints[0].set_color('r')  # Mark reference point as a red star

    # color label
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Add models to Taylor diagram
    for i, (stddev, corrcoef, name) in enumerate(samples):
        dia.add_sample(stddev, corrcoef,
                       marker='o', ms=6, ls='', alpha=1,
                       mfc=colors[i],
                       label=name)

    # Add RMS contours, and label them
    contours = dia.add_contours(levels=5, colors='0.5')  # 5 levels in grey
    plt.clabel(contours, inline=1, fontsize=10, fmt='%.0f')

    dia.add_grid()  # Add grid
    dia._ax.axis[:].major_ticks.set_tick_out(True)  # Put ticks outward

    # Add a figure legend and title
    fig.legend(dia.samplePoints,
               [p.get_label() for p in dia.samplePoints],
               numpoints=1, prop=dict(size='small'), loc='upper right')

    fig.suptitle("Model performance comparison (test set)", size='x-large')  # Figure title
    return fig

def data_overview():
    # df = pd.read_excel('data.xlsx', engine='openpyxl').iloc[:, 1:]
    df = pd.read_csv('data.csv')
    corr_matrix = df.corr().round(3)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    # plt.title('相关矩阵热力图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # df = pd.read_csv('data.csv')
    # viz(df, 'RBA replacement ratio')
    taylor_diagram(df=pd.read_csv('./performance_n/test_p.csv'))
    # data_overview()