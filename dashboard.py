import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import *
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score

st.title('Portland Trail Blazers Performance Analysis')
st.subheader('By Ammar Ash Shiddiq')


st.markdown("---")

st.subheader('Preliminaries')
"""In this project, I will be performing a simple web scraping and perform some data analysis.
Since I am a portland trailblazer fan, I want to see how well is my team last year. 
The data are taken from https://www.basketball-reference.com/ and the data are official so we don't have to worry about some data error/mistakes."""

st.subheader('Descriptive Statistic')
"""
Definition: 
a descriptive statistic is a summary statistic that quantitatively describes or summarizes 
features from a collection of information. this is what I will do for this time. I want to do
clustering and see how my favorite team performed last year, compared to previous years.
"""

st.subheader('Correlation between the data')
"Feature selection is an important step to do before you analyze a dataset. I was sure that these are the important variables: "
"W/L% = Win per Loss Percentage"
"SRS = Simple Rating System"
"Pace = An Estimate of possessions per 48 Minute"
"Rel ORtg = Relative Offensive Rating"
"Rel DRtg =  Relative Defensive Rating"
"The first step I'm going to do is to find the correlation coefficient between the variables I chose. Here's the result."


# Read the files needed
pre_df = pd.read_csv('half.csv',)
new_df = pd.read_csv('final.csv')

# Convert columns to numeric data types
pre_df['W/L%'] = pd.to_numeric(pre_df['W/L%'])
pre_df['SRS'] = pd.to_numeric(pre_df['SRS'])
pre_df['Pace'] = pd.to_numeric(pre_df['Pace'])
pre_df['Rel ORtg'] = pd.to_numeric(pre_df['Rel ORtg'])
pre_df['Rel DRtg'] = pd.to_numeric(pre_df['Rel DRtg'])



# Calculate correlation and drop any columns with missing values
data_corr = pre_df.corr(method='pearson')
data_corr = data_corr.dropna(axis=1, how='any')

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data_corr, annot=True, linewidths=.5, fmt='.1f')
st.pyplot(plt.gcf())


"""
Turns out, the correlation between pace and the other variables is considerably low. 
Therefore, we should not consider them as one of our selected variable.
"""

st.subheader('Measuring the fitness')

'Silhouette Coefficient or silhouette score is a metric used to calculate the goodness of a clustering technique. Its value ranges from -1 to 1.'

"1: Means clusters are well apart from each other and clearly distinguished."
"0: Means clusters are indifferent, or we can say that the distance between clusters is not significant."
"-1: Means clusters are assigned in the wrong way."
st. write("<a href='https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c' id='my-link'>Source</a>", unsafe_allow_html=True)

df = pre_df.drop('Pace', axis=1)
X = df

silhouette_scores = []
for i in range(2,7):
    kmeans = KMeans(n_clusters=i, n_init=10).fit(X)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)
    st.write('For value', i, 'the average silhouette score is:', silhouette_avg)

plt.figure()
plt.plot(range(2, 7), silhouette_scores, '-o')
plt.xlabel('Number of clusters, K')
plt.ylabel('Silhouette score')
st.pyplot(plt.gcf())

kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
df['cluster'] = kmeans.labels_
st.dataframe(data=df.head())



"""
A higher silhouetter score implies that the clusters are more distinct and well separated,
leading to better defined groups of data points. As we can see from our observations above, we achieved
the highest result of 0.5466484444797984 when the value is 2. This indicates that two clusters are
quite well defined with each cluster having high cohesion and decent separation to each other. 
"""

st.subheader('K-means Clustering')

"""
According to AndreyBu, who has more than 5 years of machine learning experience, stated that 
“the objective of K-means is simple: group similar data points together and discover underlying patterns. To achieve this objective, K-means looks for a fixed number (k) of clusters in a dataset.”

"""
st. write("<a href='https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1' id='my-link'>Source</a>", unsafe_allow_html=True)

"Let's look at our K-means clustering result using 2 as our our value of how many clusters we want."
st.dataframe(data=new_df.head())
"Here's the visualization."
plt.figure()
sns.pairplot(new_df, hue='cluster')
st.pyplot(plt.gcf())

"Now we know what cluster does the trailblazer 22-23 belongs to and the visualization of the two cluster. Let's see for all the years where the data is available, which cluster do they belong to."


seasons = new_df["Season"]
clusters = new_df["cluster"]

plt.figure(figsize=(10, 2))
plt.plot(seasons[::-1], clusters[::-1], marker='o', linestyle='-')
plt.xlabel("Season")
plt.ylabel("Cluster")
plt.title("Clusters over Seasons")
plt.xticks(rotation=90)
plt.grid(True)
plt.yticks([0, 1])
st.pyplot(plt.gcf())

"So there you go. This is the dashboard from the web scraping and data analysis steps that i have done. Full steps of the web scraping and data analysis steps is in the same github project as this file. Thank you!"