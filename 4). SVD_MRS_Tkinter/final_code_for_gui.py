
import numpy as np
import pandas as pd
data = pd.io.parsers.read_csv('ratings.dat', 
    names=['user_id', 'movie_id', 'rating', 'time'],
    engine='python', delimiter='::')
movie_data = pd.io.parsers.read_csv('movies.dat',
    names=['movie_id', 'title', 'genre'],
    engine='python', delimiter='::')

ratings_mat = np.ndarray(
    shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),
    dtype=np.uint8)
ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values

normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T

A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
U, S, V = np.linalg.svd(A)

def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1 # Movie id starts from 1
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

sim_movies = []

def print_similar_movies(movie_data, movie_id, top_indexes):
    print('Recommendations for {0}: \n'.format(
    movie_data[movie_data.movie_id == movie_id].title.values[0]))
    for id in top_indexes + 1:
        sim_movies.append(movie_data[movie_data.movie_id == id].title.values[0])
        
from tkinter import *        
def show_data():
    txt.delete(0.0, END)
    movie =ent.get()
    movie_title = movie        
    movie_id = movie_data[movie_data.title == movie_title].values[0][0]    
    top_n = 10
    k = 50
    sliced = V.T[:, :k]
    indexes = top_cosine_similarity(sliced, movie_id, top_n)
    print_similar_movies(movie_data, movie_id, indexes)
    print(sim_movies)
    for i in range(len(sim_movies)):
        txt.insert(1.0+i, sim_movies[i]+"\n")  
    return None   

def clear_data():
    global sim_movies
    sim_movies = []
    return txt.delete(0.0,END)
    

root=Tk()
root.geometry("420x300")
root.title("Movie Recommender System by AG-12")             
l1 = Label(root, text="Enter Movie name: ")
l2 = Label(root, text="Top Ten Suggtion For You: ")
               
ent =Entry(root)
            
l1.grid(row=0)
l2.grid(row=2)
               
ent.grid(row=0, column=1)
                                                               
txt=Text(root,width=50,height=13, wrap=WORD)
txt.grid(row=3, columnspan=2, sticky=W)
               
btn=Button(root, text="Search", bg="purple", fg="white", command=show_data)
btn1=Button(root, text="Clear", bg="purple", fg="white", command=clear_data)
btn.grid(row=1, columnspan=2)
btn1.grid(row=13, columnspan=2)
root.mainloop()
    




