import pandas as pd
import ast
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Read the CSV file
print("reading csv...")
df = pd.read_csv("data_original_v2.csv")

# ['url', 'name', 'owner', 'forks', 'watchers', 'stars', 'languages', 'commits', 'creation_date', 'contributors', 'topics', 'subscribers', 'readme', 'releases', 'pull_requests', 'readme_size', 'commits_freq', 'releases_freq', 'lines_of_codes', 'popularity_score_1', 'popularity_score_2', 'popularity_score_3']


# Define keywords to search for
keywords = ["Artificial Intelligence", "Machine Learning", "Deep Learning", "Neural Network", "Natural Language Processing", "Computer Vision", "Explainable AI", "Convolutional Neural Network", "Transformer", "Diffusion", "Language Model"]


# Load pre-trained Sentence-BERT model
print("loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentenceTransformer("all-MiniLM-L6-v2")
model = model.to(device)


# Encode keywords into embeddings
keyword_embeddings = model.encode(keywords, normalize_embeddings=True)

# Function to check if any topic word is similar to the keywords
def is_similar(topics_str, keyword_embeddings, threshold=0.7):
    try:
        topics_list = ast.literal_eval(topics_str)  # Convert string to list
        if not isinstance(topics_list, list) or len(topics_list) == 0:  # Ignore empty lists
            return False
    except:
        return False  # If parsing fails, treat as no match

    # Encode each topic word
    topic_embeddings = model.encode(topics_list, normalize_embeddings=True)

    # Compute cosine similarity between each topic word and keywords
    similarity_matrix = cosine_similarity(topic_embeddings, keyword_embeddings)

    # Check if any word in topics has similarity > threshold
    return (similarity_matrix.max(axis=1) > threshold).any()

# Apply the function to filter rows
df["is_matched"] = df["topics"].apply(lambda x: is_similar(str(x), keyword_embeddings))

# Create two separate dataframes
matched_df = df[df["is_matched"]].drop(columns=["is_matched"])  # Rows where similarity > threshold
unmatched_df = df[~df["is_matched"]].drop(columns=["is_matched"])  # Rows where no match found

# Display results
print("Matched Rows:")
print(matched_df['topics'])

# print("\nUnmatched Rows:")
# print(unmatched_df)

# Optionally save the results
matched_df.to_csv("AI_repos.csv", index=False)
unmatched_df.to_csv("nonAI_repos.csv", index=False)


# matched_df = pd.read_csv("AI_repos.csv")
# unmatched_df = pd.read_csv("nonAI_repos.csv")

# print("unmatched instances: {}    stars: {} {}".format(len(unmatched_df), unmatched_df["stars"].mean(), unmatched_df["stars"].std()))
# print("matched instances: {}    stars: {} {}".format(len(matched_df), matched_df["stars"].mean(), matched_df["stars"].std()))
