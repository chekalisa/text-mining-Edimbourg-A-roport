import matplotlib.pyplot as plt
import pandas as pd

def histograms_percent(df, columns):
    plt.figure(figsize=(18, 10))
    for i, column in enumerate(columns, 1): # enumerate to iterate over a sequence while keeping track of the index of each element.
        plt.subplot(3, 3, i)
        response_counts = df[column].value_counts(normalize=True) * 100  # Calculate percentages
        response_counts.plot(kind='bar', color='skyblue')
        plt.xlabel('Responses')
        plt.ylabel('Percentage')
        plt.title(column)
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        for index, value in enumerate(response_counts):
            plt.text(index, value, f'{value:.1f}%', ha='center', va='bottom') #f'{value:.1f} formats the value to one decimal place.
    plt.tight_layout()
    plt.show()
#'center' means the text will be horizontally centered with respect to the x-coordinate (index).
#'bottom' means the text will be aligned to the bottom of the y-coordinate (value), placing the text just above the bar.

def time(df):

    df["time"]=df["Start Date (UTC)"].dt.date
    data_count = df.groupby(pd.Grouper(key='Start Date (UTC)', freq='M')).size()

    # Plot the results
    plt.figure(figsize=(14, 7))
    data_count.sort_index().plot(kind='line', marker='o')
    plt.title('Number of Reviews over time')
    plt.xlabel("Date")
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=45)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_vader_suggestions(df):
    df['Start Date (UTC)'] = pd.to_datetime(df['Start Date (UTC)'])
    df['month'] = df['Start Date (UTC)'].dt.to_period('M').astype(str)
    monthly = df.groupby('month')['sentiment_suggestions'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(monthly['month'], monthly['sentiment_suggestions'], marker='o', linestyle='-')
    plt.xlabel('Month')
    plt.ylabel('Average Sentiment Score')
    plt.title('Average Sentiment Score of Feedbacks by Month')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_vader_services(df):
    df['Start Date (UTC)'] = pd.to_datetime(df['Start Date (UTC)'])
    df['month'] = df['Start Date (UTC)'].dt.to_period('M').astype(str)
    monthly = df.groupby('month')['sentiment_services'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(monthly['month'], monthly['sentiment_services'], marker='o', linestyle='-')
    plt.xlabel('Month')
    plt.ylabel('Average Sentiment Score')
    plt.title('Average Sentiment Score of Premium Services suggested by Month')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_vader_agg(df):
    df['Start Date (UTC)'] = pd.to_datetime(df['Start Date (UTC)'])
    monthly = df.groupby('month')['combined_sentiment'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(monthly['month'], monthly['combined_sentiment'], marker='o', linestyle='-')
    plt.xlabel('Month')
    plt.ylabel('Average Sentiment Score')
    plt.title('Average Sentiment Score by Month')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_vader_aspect(df):
    df.plot(kind='bar', title='Average Sentiment for Each Aspect')
    plt.xlabel('Aspect')
    plt.ylabel('Average Sentiment')
    plt.xticks(rotation=45)
    plt.show()