import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Portland Trail Blazers Performance Analysis')
st.subheader('By Ammar Ash Shiddiq')


st.markdown("---")

st.subheader('Preliminaries')
"""I will be performing a simple web scraping and perform some data analysis.
Since I am a portland trailblazer fan, I want to see how well is my team last year. 
The data are taken from https://www.basketball-reference.com/. 
The data are official so we don't have to worry about some data error/mistakes."""

st.subheader('Understanding the Problem')
"""
A descriptive statistic is a summary statistic that quantitatively describes or summarizes 
features from a collection of information. this is what I will do for this time. I want to do
clustering and seeing how my favorite team performed last year, compared to previous years
"""

st.subheader('Correlation between the data')
"Feature selection is an important step to do before you perform data clustering. I was sure that these are the important variables: "
"W/L% = Win per Loss Percentage"
"SRS = Simple Rating System"
"Pace = An Estimate of possessions per 48 Minute"
"Rel ORtg = Relative Offensive Rating"
"Rel DRtg =  Relative Defensive Rating"
"So I find the correlation coefficient between the variables I chose and here's the result"
