import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def load_movies(csv_path='movies.csv'):
    df = pd.read_csv(csv_path)
    df['plot']=df['plot'].fillna("").astype(str).str.strip()
    df['title']=df['title'].astype(str).str.strip()
    return df

def load_model(model_dir):
    return SentenceTransformer(model_dir)

def load_embeddings(path):
    return np.load(path)

def search(query, model, embeddings, df, top_n=5):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores = (q_emb @ embeddings.T).ravel()
    k = min(top_n, len(scores))
    idxs = np.argpartition(-scores, range(k))[:k]
    idxs = idxs[np.argsort(-scores[idxs])]
    out = df.iloc[idxs].copy().reset_index(drop=True)
    out.insert(0, 'rank', range(1, len(out)+1))
    out['score'] = scores[idxs]
    return out[['rank','title','plot','score']]