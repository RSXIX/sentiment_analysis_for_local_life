# Performing Sentiment Analysis on IMDB Data Set

This project enables the user to run sentiment analysis on a data set from Hugging Face using [Twitter-Roberta-Base-Sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment). A subset of the data set (the training split) will be used to test the machine learning model - each of the reviews in the data set will be categorized as either positive, neutral or negative.

## Data Set

This [dataset](https://huggingface.co/datasets/imdb) was used for the project. It consists of a series of user reviews of different movies on IMDb.

## Installation

1. Clone the repository to your computer.
1. You may check on the CLI if you have python installed on your system using `python --version`. If it is not there, or if it is not Python 3.X, then navigate to the official [website](https://www.python.org/downloads/) and download version 3.11.5.
1. If virtual environment is not already installed on system, you may do so using `pip install virtualenv`
1. Initialize a python virtual environment for the project using `python3 -m venv venvlocallife`
1. Active the virtual environment using `source venvlocallife/bin/activate`
1. Install the required dependencies for the project in the virtual environment using `pip install -r requirements.txt`
1. You may now open the file `sentiment_analysis.py` and run the code. This will be a lengthy process as we initially use all the reviews in the data set. Check the next section if you want to reduce the amount of examples being used.
1. Once done, you may deactivate the virtual enviornment and go back to your local python environment using `deactivate`.

### Splitting the data further

The dataset in `sentiment_analysis.py` originally has 25000 reviews to load, tokenize, train and evaluate. It is possible to reduce the amount of reviews being processed to an arbitary amount of reviews using the following code:

```
#Helper function used to select the first x and last y elements from the total dataset
def select_elements(df, x,y):
    first_x = df.head(x)
    last_y = df.tail(y)
    df = pd.concat([first_x, last_y])
    return df
```

In the `sentiment_analysis.py` file, the first and last 50 values of the data set are used. Instructions are also provided on how to perform sentiment analysis for the entire dataset



