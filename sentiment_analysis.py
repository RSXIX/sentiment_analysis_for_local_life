from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
import pandas as pd


dataset = load_dataset("imdb", split = "train")

dataframe = pd.DataFrame(dataset)

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def model_score(text):
    #Tokenization
    tokens = tokenizer.encode(text, return_tensors='pt')
    #Running the model on given review and evaluating the sentiment
    result = model(tokens)
    scores = result[0][0].detach().numpy()
    scores = softmax(scores)
    max_sentiment_index = np.argmax(scores)
    if max_sentiment_index == 0:
        sentiment = "Negative"
    elif max_sentiment_index == 1:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"
    return sentiment

runTimeErrorCount = 0
indexErorCount = 0

#Helper function used to select the first x and last y elements from the total dataset
def select_elements(df, x,y):
    first_x = df.head(x)
    last_y = df.tail(y)
    df = pd.concat([first_x, last_y])
    return df

df = select_elements(dataframe, 50,50) # Ammend the x and y value to slice the dataset as you wish
# Alternatively you can uncomment the following line and comment the prior line to perform sentiment analysis on the entire dataset:
# df = dataframe


print(df)

#Upon going through the model and the errors, 
#It was evident they arose due to some sentences exceeding the maximum character count of the model we are using
for i, example in df.iterrows():
    try:
        review = df.at[i,'text']
        score = model_score(review)
        df.at[i, 'Sentiment'] = model_score(review)
    except RuntimeError as err:
        review = df.at[i,'text']
        score = model_score(review[:512])
        runTimeErrorCount += 1
        df.at[i, 'Sentiment'] = score
    except Exception as e:
        indexErorCount = 0
        df.at[i, 'Sentiment'] = "Could not find the sentiment!"
    
print(df)






