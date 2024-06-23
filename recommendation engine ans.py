
'''
# Data Mining Unsupervised Learning / Descriptive Modeling - Recommendation engine

IT IS PERSONALIZED CONCEPT:

# Problem Statement:
    
Q1) Build a recommender system with the given data using UBCF.
        
        This dataset is related to the video gaming industry and a survey was conducted to build a 
        recommendation engine so that the store can improve the sales of its gaming DVDs. A snapshot 
        of the dataset is given below. Build a Recommendation Engine and suggest top-selling DVDs to the store customers.



# CRISP-ML(Q) process model describes six phases:
# 
# 1. Business and Data Understanding
# 2. Data Preparation
# 3. Model Building
# 4. Model Evaluation
# 5. Deployment
# 6. Monitoring and Maintenance
'''





'''
1st STEP:
1. Business and Data Understanding :
'''
# Objective(s): increase the sales of its gaming DVDs. 
# Constraints : reduce the cost for marketting compaigns and reduce the cost of low demand gaming advertisement


'''Success Criteria'''

# Business Success Criteria: satisfy customer is more likely to become a loyal customer 
#                            leading to increase a revenue over time.

# ML Success Criteria:       

# Economic Success Criteria: increasing more sales through personalized recommendations,
#                            the store can experience a significant boost in revenue. 


'''
data collection
'''
# dataset of gaming industry is availbale in our lMS website.
# this dataset is included various games with different different user ratings.
# rows - dataset contain 5000 users who ratings video games (range like from 1.0 to 5.0)  
# col  - and 3 columns which contain user id, name for game and ratings for each users.
                   
# data description : 

# userId : An identifier for each user who rated the games.
# game   : The name of the video game .
# rating : The user's rating for the respective game on a 
#          scale from 1.0 to 5.0, with 1.0 being the lowest rating and 5.0 being the highest rating.    
        
  
'''
2nd STEP: 
Data preparation (data cleaning)    
'''
import pandas as pd
game = pd.read_csv(r"D:\DATA SCIENTIST\DATA SCIENCE\DATASETS\game.csv")


# Credentials to connect to sql Database
from sqlalchemy import create_engine
user = 'root'  # user name
pw = 'root'  # password
db = 'game_db'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# to_sql() - function to push the dataframe onto a SQL table.
game.to_sql('game_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql = 'select * from game_tbl;'
df = pd.read_sql_query(sql, engine) 



df.shape
df.dtypes
df.info()
df.describe()
df.duplicated().sum()
# does not present any duplicate records in our dataset

# now i want to check if any outliers are present or not 
# outlier treatment:
df.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
# here i see some outliers are present in rating column.
# now i want to replace outliers as a inliers using winsorizer


from feature_engine.outliers import Winsorizer 
winsor = Winsorizer(capping_method = 'iqr',
                    fold = 1.5 ,
                    tail = 'both',
                    variables = ['rating']) 


df['rating'] = winsor.fit_transform(df[['rating']])
import seaborn as sns
sns.boxplot(df['rating']) # here we replace outliers to inliers from rating column

df.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 

# our dataset is clean  


# ========================================================================================================
# =------------------------------------------------------=================-===------------------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

     
  
'''
3rd STEP: 
Model Building (data mining)    
'''



# Create a Tfidf Vectorizer to remove all stop words
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 


tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfidf vectorizer 

# Transform a count matrix to a normalized tf-idf representation
tfidf_matrix = tfidf.fit(df.game)   



# Save the Pipeline for tfidf matrix
import joblib
joblib.dump(tfidf_matrix, 'games')  

import os 
os.getcwd()

# Load the saved model for processing
mat = joblib.load("games") 

tfidf_matrix = mat.transform(df.game) 

tfidf_matrix.shape   

# cosine(x, y)= (x.y) / (||x||.||y||)
# Computing the cosine similarity on Tfidf matrix

cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a mapping of dataframe name to index number
df_index = pd.Series(df.index, index = df['game']).drop_duplicates() 

# Example
df_id = df_index['SoulCalibur'] 
df_id

topN = 15





# Custom function to find the TopN games to be recommended

def get_recommendations(userId, topN):    
    # topN = 10
    # Getting the movie index using its title 
    df_id = df_index[userId]
    
    # Getting the pair wise similarity score for all the games with that games
    cosine_scores = list(enumerate(cosine_sim_matrix[df_id])) 
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key = lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN + 1]
    
    # Getting the movie index 
    df_idx  =  [i[0] for i in cosine_scores_N]
    df_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar games and scores
    similar_games = pd.DataFrame(columns = ["userId", "Score"])
    similar_games["userId"] = df.loc[df_idx, "userId"]
    similar_games["Score"] = df_scores
    similar_games.reset_index(inplace = True)  
    # similar_games.drop(["index"], axis=1, inplace=True)
    return(similar_games.iloc[1:, ])

# Call the custom function to make recommendations
rec = get_recommendations(10, topN = 25) 
rec  


'''
on the basis of given output we can say - the users whoose score is equal to 1 
they are more likely to purchased a recommended DVDs. this is the strong signal for
these DVDs align closely to users interest, therefore business can expect increase in sales
for the DVDs with high recommendation score, also increase in revenue.

'''


























