#pip install ollama
import os
import ollama
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sentence1 = "I like maths"
sentence2 = "I like science"
sentence3 = "I like sugar"

response = ollama.embed(
  model="mxbai-embed-large",
  input=sentence1
)
response1 = ollama.embed(
  model="mxbai-embed-large",
  input=sentence2
)
response2 = ollama.embed(
  model="mxbai-embed-large",
  input=sentence3
)
embed1 = np.array(response['embeddings']).reshape(1, -1)
embed2 = np.array(response1['embeddings']).reshape(1, -1)
embed3 = np.array(response2['embeddings']).reshape(1, -1)

print(cosine_similarity(embed1,embed2))
print(cosine_similarity(embed2,embed3))
print(cosine_similarity(embed3,embed1))
