# Semantic Search System with Semantic Cache

## Project Overview

This project implements a lightweight semantic search system on the 20 Newsgroups dataset (~20k documents).
It combines vector embeddings, fuzzy clustering, and a semantic cache to retrieve semantically similar documents while avoiding repeated computation for similar queries.

The system is exposed through a FastAPI service.

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

```
User Query
   ↓
Query Embedding
   ↓
Semantic Cache Lookup
   ↓
 ┌───────────────┐
 │   Cache Hit   │ → Return Cached Result
 └───────────────┘
        │
        ▼
 ┌───────────────┐
 │  Cache Miss   │
 └───────────────┘
        ↓
FAISS Vector Search
        ↓
Cluster Identification
        ↓
Store Result in Cache
        ↓
Return Result
```
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

## Running with Docker

Build the container:

```
docker build -t semantic-search .
```

Run the container:

```
docker run -p 8000:8000 semantic-search
```

Or using docker-compose:

```
docker-compose up --build
```

Open the API documentation:

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
