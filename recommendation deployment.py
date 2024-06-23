from flask import Flask, render_template, request
import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity

user = 'root'  # user name
pw = 'root'  # password
db = 'game_db'  # database name

engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

sql = 'select * from game_tbl;'
df = pd.read_sql_query(sql, engine)


mat = joblib.load("games") 

tfidf_matrix = mat.transform(df.game) 

cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

df_index = pd.Series(df.index, index = df['game']).drop_duplicates() 





# Custom function to find the TopN movies to be recommended

def get_recommendations(game, topN):    
    # topN = 10
    # Getting the movie index using its title 
    df_id = df_index[game]
    
    # Getting the pair wise similarity score for all the anime's with that anime
    cosine_scores = list(enumerate(cosine_sim_matrix[df_id])) 
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key = lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN + 1]
    
    # Getting the movie index 
    df_idx  =  [i[0] for i in cosine_scores_N]
    df_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    anime_similar_show = pd.DataFrame(columns = ["game", "Score"])
    anime_similar_show["game"] = df.loc[df_idx, "game"]
    anime_similar_show["Score"] = df_scores
    anime_similar_show.reset_index(inplace = True)  
    # anime_similar_show.drop(["index"], axis=1, inplace=True)
    return(anime_similar_show.iloc[1:, ])

######End of the Custom Function######    

app = Flask(__name__)

@app.route('/')
def home():
    #colours = ['Red', 'Blue', 'Black', 'Orange']
    return render_template("index.html")

@app.route('/guest', methods = ["post"])
def Guest():
    if request.method == 'POST' :
        mn = request.form["mn"]
        tp = request.form["tp"]
        
        top_n = get_recommendations(mn, topN = int(tp))

        # Transfering the file into a database by using the method "to_sql"
        top_n.to_sql('top_10', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
        
        html_table = top_n.to_html(classes = 'table table-striped')

        return render_template( "data.html", Y = "Results have been saved in your database", Z =  f"<style>\
                    .table {{\
                        width: 50%;\
                        margin: 0 auto;\
                        border-collapse: collapse;\
                    }}\
                    .table thead {{\
                        background-color: #39648f;\
                    }}\
                    .table th, .table td {{\
                        border: 1px solid #ddd;\
                        padding: 8px;\
                        text-align: center;\
                    }}\
                        .table td {{\
                        background-color: #5e617d;\
                    }}\
                            .table tbody th {{\
                            background-color: #ab2c3f;\
                        }}\
                </style>\
                {html_table}") 

if __name__ == '__main__':

    app.run(debug = True)  