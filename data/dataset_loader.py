from sklearn.datasets import fetch_20newsgroups


def load_dataset():

    """
    Load and preprocess the 20 Newsgroups dataset for semantic analysis.

    Preprocessing decisions:
    - Remove headers, footers, and quotes because they contain metadata
      (email signatures, reply chains) that do not contribute to semantic meaning.
    - Remove very short documents because they provide weak semantic signals
      and introduce noise in embedding space.
    """

    # Fetch dataset
    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes")  # remove noisy metadata
    )

    documents = dataset.data

    # Remove extremely short documents to reduce noise in embeddings
    documents = [doc for doc in documents if len(doc.strip()) > 50]

    return documents