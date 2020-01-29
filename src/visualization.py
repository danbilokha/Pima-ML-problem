import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go


def target_count(data):
    """Count target variable.

    Arguments
    data : Pandas DataFrame
        Dataset frame with features and target variable.

    """
    trace = go.Bar(
        x= data['Outcome'].value_counts().values.tolist(),
        y=['healthy','diabetic' ],
        orientation='h',
        text=data['Outcome'].value_counts().values.tolist(),
        textfont=dict(size=15),
        textposition='auto',
        opacity=0.8,
        marker=dict(color=['lightskyblue', 'gold'],
                    line=dict(color='#000000',width=1.5))
    )
    layout = dict(title='Count of Outcome variable')

    fig = dict(data=[trace], layout=layout)
    py.iplot(fig)


def target_percent(data):
    """Show in percenteges counts.

    Arguments
    data : Pandas DataFrame
        Dataset frame with features and target variable.

    """
    trace = go.Pie(
        labels=['healthy','diabetic'],
        values=data['Outcome'].value_counts(),
        textfont=dict(size=15),
        opacity=0.8,
        marker=dict(colors=['lightskyblue', 'gold'],
                    line=dict(color='#000000', width=1.5))
    )
    layout = dict(title='Distribution of Outcome variable')

    fig = dict(data=[trace], layout=layout)
    py.iplot(fig)


def missing_plot(dataset, key):
    """Define missing plot to detect all missing values in dataset.

    Arguments
    data : Pandas DataFrame
        Dataset frame with features and target variable.
    key : string
        Key feature to look for.

    """
    null_feat = pd.DataFrame(len(dataset[key]) - dataset.isnull().sum(), columns=['Count'])
    percentage_null = pd.DataFrame(
        (len(dataset[key]) - (len(dataset[key]) - dataset.isnull().sum())) / len(dataset[key]) * 100,
        columns=['Count'])
    percentage_null = percentage_null.round(2)

    trace = go.Bar(
        x=null_feat.index,
        y=null_feat['Count'],
        opacity=0.8,
        text=percentage_null['Count'],
        textposition='auto',
        marker=dict(color='#7EC0EE',
                    line=dict(color='#000000', width=1.5))
    )
    layout = dict(title="Missing Values (count & %)")

    fig = dict(data=[trace], layout=layout)
    py.iplot(fig)


def correlation_plot(data):
    """Compute correlation matrix and visualise it.

    Arguments
    data : Pandas DataFrame
        Dataset frame with features and target variable.

    """
    correlation = data.corr()
    matrix_cols = correlation.columns.tolist()
    corr_array  = np.array(correlation)
    trace = go.Heatmap(
        z=corr_array,
        x=matrix_cols,
        y=matrix_cols,
        colorscale='Viridis',
        colorbar=dict(),
    )
    layout = go.Layout(
        dict(title='Correlation Matrix for variables',
             margin=dict(r=0, l=100,
                         t=0, b=100,),
             yaxis=dict(tickfont=dict(size=9)),
             xaxis=dict(tickfont=dict(size=9)),)
    )
    fig = go.Figure(data = [trace],layout = layout)
    py.iplot(fig)
