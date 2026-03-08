# Semantic Search System with Semantic Cache

## Project Overview

This project implements a lightweight semantic search system using the 20 Newsgroups dataset.  
The system performs semantic document retrieval using vector embeddings, fuzzy clustering, and a semantic caching mechanism.

## Features

- Semantic search using transformer embeddings
- Vector database using FAISS
- Fuzzy clustering using Gaussian Mixture Models
- Semantic cache for similar queries
- FastAPI REST API service

## Dataset

20 Newsgroups dataset (~20,000 news posts)

Loaded automatically using:

```
sklearn.datasets.fetch_20newsgroups
```

## System Architecture

User Query  
↓  
Embedding Model  
↓  
Semantic Cache Check  
↓  
Vector Database Search  
↓  
Cluster Analysis  
↓  
Return Results  

## Tech Stack

Python  
FastAPI  
Sentence Transformers  
FAISS  
Scikit-learn  

## API Endpoints

### POST /query

Input:

```json
{
 "query": "space exploration and nasa missions"
}
```

Output:

```
{
 "query": "...",
 "cache_hit": false,
 "matched_query": "...",
 "similarity_score": 0.91,
 "result": "...",
 "dominant_cluster": 3
}
```

### GET /cache/stats

Returns cache statistics.

### DELETE /cache

Clears the cache.

## Setup Instructions

1. Clone the repository

```
git clone <repo-url>
cd semantic-cache-system
```

2. Create virtual environment

```
python -m venv venv
```

3. Activate environment

Windows

```
venv\Scripts\activate
```

4. Install dependencies

```
pip install -r requirements.txt
```

5. Run the API

```
uvicorn api.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

## Design Decisions

Embedding Model:
- all-MiniLM-L6-v2 (lightweight and efficient)

Vector Database:
- FAISS for fast similarity search

Clustering:
- Gaussian Mixture Model for soft clustering

Cache Strategy:
- cosine similarity between query embeddings

## Author

Dheerendra Pratap Singh