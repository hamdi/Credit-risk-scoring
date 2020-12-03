import matplotlib.pyplot as plt
from matplotlib import collections as mc
import seaborn as sns
import plotly.graph_objects as go
from plotly.offline import iplot
import plotly.figure_factory as ff
from plotly import subplots
import plotly.express as px
import pandas as pd
import numpy as np

def plot_variables(labels, plot, data):
    """
    Plot individual variables with dropdown menu selection
    param plot: One of the 3 plot types supported: 0 for barplot, 1 for histogram, 2 for boxplot
    param labels: iterable containing the variable names
    """
    # Create individual figures
    fig = subplots.make_subplots(rows=1, cols=1)
    for var in labels:
        if plot == 0:
            counts = data[var].value_counts()
            fig.append_trace(go.Bar(x=counts, y=counts.index, orientation='h'), 1, 1)
        elif plot == 1:
            fig.append_trace(ff.create_distplot([list(data[var])], ['distplot'])['data'][0], 1, 1)
            fig.append_trace(ff.create_distplot([list(data[var])], ['distplot'])['data'][1], 1, 1)
        elif plot == 2:
            fig.add_trace(go.Box(x=list(data[data["Score"] == "good"][var]), name="Good", hoverinfo="x", marker_color='mediumturquoise'))
            fig.add_trace(go.Box(x=list(data[data["Score"] == "bad"][var]), name="Bad", hoverinfo="x", marker_color='darkorange'))
        else:
            raise ValueError("plot number must be 0, 1, or 2")
    # Create buttons for drop down menu
    buttons = []
    for i, label in enumerate(labels):
        if plot == 0:
            visibility = [i == j for j in range(len(labels))]
        else:
            visibility = [j//2 == i for j in range(2*len(labels))]
        button = dict(
            label=label,
            method='update',
            args=[{'visible': visibility},
                  {'title': label}])
        buttons.append(button)
    updatemenus = list([
        dict(active=-1,
             x=1.06, y=1.27,
             buttons=buttons
             )
    ])
    # Setup layout
    if plot == 0:
        fig['layout']['title'] = "Distribution of categorical and discrete variables:"
        fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                          marker_line_width=1.5, opacity=0.7)
    elif plot == 1:
        fig['layout']['title'] = "Distribution of continuous variables:"
        fig.update_traces(marker_color='rgb(112, 125, 188)', opacity=0.8)
    elif plot == 2:
        fig['layout']['title'] = "Boxplot of continuous variables by score:"
    fig['layout']['showlegend'] = False
    fig['layout']['updatemenus'] = updatemenus
    iplot(fig, config={"displayModeBar": False})


def plot_pies(labels, data, categories):
    """
    Plot nested pies of the good/bad distribution for each category with variable selection from dropdown menu
    param labels: iterable containing the variable names
    """
    # Create the figure and traces
    fig = go.Figure()
    visibility, annotations = [], []
    traces = 0
    for var in labels:
        if var in categories:
            values = list(dict.fromkeys(categories[var].values()))
        else:
            values = list(dict.fromkeys(data[var]))
            values.sort()
        nvals = len(values)
        counts = []
        annotations.append([])
        for i in values: # get the counts for each category
            s = data[(data[var] == i) & (data["Score"] == "good")].shape[0]
            counts.append([s, data[data[var] == i].shape[0]-s])
        for i in range(nvals): # plot them in nested pie charts
            fig.add_trace(go.Pie(hole=(0.3+i*0.7/nvals)/(0.3+(0.7*(i+1)/nvals)), sort=False,
                                 domain={'x': [0.35-(i+1)*0.35/nvals, 0.65+(i+1)*0.35/nvals], 'y': [0.35-(i+1)*0.35/nvals, 0.65+(i+1)*0.35/nvals]}, name=values[i], values=counts[i], textposition='inside',
                                 textfont_size=20, marker={'colors': ['mediumturquoise', 'darkorange'], 'line': {'color': '#000000', 'width': 2}},
                                 labels=['Good', 'Bad'], hoverinfo='label+name+percent', textinfo='percent'))
            annotations[-1].append(dict(text=values[i], x=0.577+(i+1)*0.184/nvals, y=0.5,
                                        arrowwidth=1.5, arrowhead=6, ax=100+20*i, ay=100+20*i))
            traces += 1
        visibility = [l+nvals*[False] for l in visibility]
        visibility.append((traces-nvals)*[False]+nvals*[True])
    # Create buttons for drop down menu
    buttons = []
    for i, label in enumerate(labels):
        button = dict(
            label=label,
            method='update',
            args=[{'visible': visibility[i]},
                  {'title': label, 'annotations': annotations[i]}])
        buttons.append(button)
    updatemenus = list([
        dict(active=-1,
             x=1.06, y=1.27,
             buttons=buttons)])
    # Setup layout
    fig['layout']['updatemenus'] = updatemenus
    fig['layout']['title'] = "Distribution of the scores of categorical variables:"
    iplot(fig, config={"displayModeBar": False})

def plot_importance_metrics(feat_importance, variables):
    sorted_idx = feat_importance["Average"].argsort()
    feat_importance=pd.melt((feat_importance*10).T[sorted_idx].T.reset_index(),id_vars='index')
    feat_importance["metric"]=feat_importance["variable"]
    feat_importance["variable"]=[variables[i] for i in feat_importance["index"]]
    fig, ax = plt.subplots(figsize=(5,10), dpi=200)
    sns.stripplot(data=feat_importance, hue="metric", y="variable", x="value")
    lines = ([[n, y] for n in group] for y, (_, group) in enumerate(feat_importance.groupby(['variable'], sort = False)['value']))
    lc = mc.LineCollection(lines, colors='lightgrey', linewidths=1)
    ax.add_collection(lc)
    ax.set_title("Feature Importance indicators (rescaled)")
    fig.tight_layout()
    plt.show()
    
    
def plot_confusion_matrix(confusion_matrix, ax=None, title=""):
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    df_cm = pd.DataFrame(confusion_matrix, index=["Bad", "Good"], columns=["Bad", "Good"])
    fig = plt.figure()
    strings = np.asarray([['True Negatives', 'False Positives'],
                          ['False Negatives', 'True Positives']])
    labels = np.asarray(
        [f"{value:d}\n{string}" for string, value in zip(
            strings.flatten(), confusion_matrix.ravel())]).reshape(2, 2)
    sns.heatmap(df_cm, annot=labels, fmt="", cmap=cmap, ax=ax).set_title(title)
    if ax:
        ax.set_xlabel('Predictions')
        ax.set_ylabel('References')
        plt.close()
    else:
        plt.ylabel('References')
        plt.xlabel('Predictions')
        plt.close()
        return fig

def plot_ROC(results):
    traces = [go.Scatter(x=[0, 1], y=[0, 1],
                     mode='lines',
                     line=dict(color='navy', width=1, dash='dash'),
                     showlegend=False)]
    for i in results.index:
        trace = go.Scatter(x=results.loc[i]['fpr'], y=results.loc[i]['tpr'],
                        mode='lines',
                        line=dict(width=1),
                        name=f"{i} (AUC = {results.loc[i]['auc']:0.4f}\nCost = {results.loc[i]['cost']:0.4f})"
                        )
        traces.append(trace)
    layout = go.Layout(title='Receiver Operating Characteristic curve',
                   xaxis=dict(title='False Positive Rate'),
                   yaxis=dict(title='True Positive Rate'))

    fig = go.Figure(data=traces, layout=layout)
    return fig

def plot_votes(class1, class2, names):
    N = names.shape[0]
    ind = np.arange(N) 
    width = 0.2
    fig, ax = plt.subplots(figsize=(8,4), dpi=200)
    p1 = ax.bar(ind, np.hstack(([class1[:-1], [0]])), width,
                color='red', edgecolor='k')
    p2 = ax.bar(ind + width, np.hstack(([class2[:-1], [0]])), width,
                color='steelblue', edgecolor='k')
    p3 = ax.bar(ind, [0, 0, 0, 0, class1[-1]], width,
                color='darkgreen', edgecolor='k')
    p4 = ax.bar(ind + width, [0, 0, 0, 0, class2[-1]], width,
                color='lightgreen', edgecolor='k')
    plt.axvline(3.7, color='k', linestyle='dashed')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(names,
                    rotation=40,
                    ha='right')
    plt.ylim([0, 1])
    plt.title('Class probabilities for one sample by different classifiers')
    plt.legend([p1[0], p2[0]], ['Bad', 'Good'], loc='upper left')
    plt.tight_layout()

    plt.show()