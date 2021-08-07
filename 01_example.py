import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

st.text("TEST TEXT")
st.write(1234)
df = pd.DataFrame(
    {'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40],}
    )

st.text("Using Magic")
df

st.text("Using st.write")
st.write(df)


st.write('Below is a DataFrame:', df, 'Above is a dataframe.')

# import altair as alt

# df = pd.DataFrame(np.random.randn(200, 3),columns=['a', 'b', 'c'])
# c = alt.Chart(df).mark_circle().encode(x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
# st.write(c)

st.text("LINE PLOT USING")
from bokeh.plotting import figure

# x = np.arange(100)
# y = x ** 2
x = [1, 2, 3, 4, 5]
y = [2, 1, 4, 1, 6]

fig = figure(title="Test Plot",
            x_axis_label="x",
            y_axis_label="y")

fig.line(x, y, legend_label="Line", line_width=2)
st.bokeh_chart(fig, use_container_width=True)

st.text("LINE CHART USING ST")
chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

st.text("MAP USING ST")
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    chart_data


option = st.selectbox(
    'Which number do you like best?',
     ["mobilenet", "resnet50", ""])

'You selected: ', option

