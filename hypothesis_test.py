import re
import ast
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Read the CSV file
print("reading csv...\n")
df = pd.read_csv("github_repos.csv")

print("github_repos size: {}\n".format(len(df)))

# ['url', 'name', 'owner', 'forks', 'watchers', 'stars', 'languages', 'commits', 'creation_date', 'contributors', 'topics', 'subscribers', 'readme', 'releases', 'pull_requests', 'readme_size', 'commits_freq', 'releases_freq', 'lines_of_codes', 'popularity_score_1', 'popularity_score_2', 'popularity_score_3']


# Define keywords to search for
keywords = ["Artificial Intelligence", "Machine Learning", "Deep Learning", "Supervised Learning", "Unsupervised Learning", 
            "Reinforcement Learning", "Policy Gradient", "Autonomous Driving", "Neural Network", "Gradient Descent", "Backpropagation", 
            "Natural Language Processing", "Computer Vision", "Explainable AI", "Convolutional Neural Network", "Long Short Term Memory", 
            "Recurrent Neural Network", "Sequence to Sequence", "Seq2Seq", "Multi Layer Perceptron", "Transformer", "Diffusion", "Language Model", 
            "BERT", "Big Data", "Semantic Segmentation", "Object Tracking", "Time Series Prediction", "Contrastive Learning", "ChatGPT", 
            "Data Science", "Support Vector Machine", "Graphic Convolutional Network", "Embeddings", "Pretrained", "Pretrained Models", 
            "ResNet", "VGG", "Finetune", "Word2Vec", "FastText", "Hyperparameter Tuning", "Pytorch", "TensorFlow", "Keras", "CUDNN", 
            "Text to Speech", "Image Generation", "Image Edition", "Deep Reinforcement Learning", "ImageNet", "Image Inpainting", "Image to Image", 
            "Image Completion", "Inception", "ResNext", "Linear Regression", "LLaVa", "LLaMa", "Multi Modal", "Vision Language", "GPT", "CNN, "
            "Domain Adaptation", "Super Resolution", "Segmentation", "Tokenization", "Face Recognition", "OpenAI", "Principal Component (PCA)",
            "CUDA", "Text Detection", "Text Recognition", "Chatbot", "Video Generation", "Text Summarization", "Text Classification",
            "Internet of Things", "NLP", "Style Transfer", "Graph Neural Network", "Motion Detection", "Emotion Analysis", "Emotion Detection", 
            "Emotion Recognition", "Facial Expression", "Sentiment Analysis", "Voice Segmentation", "Semantic Scene Understanding",
            "Video Understanding", "Scene Understanding", "Natural Language Understanding", "MultiModal", "Multimodal Learning", "Multimoda Research",
            "Recommender System", "DeepFake", "DeepFake Detection", "Multi Label Classification", "Point Cloud", "Point Cloud Segmentation",
            "Point Cloud Detection", "Point Cloud Processing", "Point Cloud Segmentation", "Image Classification", "Vision Transformer",
            "Medical Image Classification", "Transfer Learning", "Fine Grained Classification", "Fine Grained Recognition",
            "Fine Grained Visual Categorization", "Hyperspectral Image Classification", "Image Processing", "Fake News Classification",
            "Text Generation", "Binary Classification", "Sentiment Classification", "Unet", "Unet Classification", "Cifar10", "MNIST",
            "Decision Tree", "Iris Classification", "Classification Algorithm", "Multi Label Learning", "Age Detection", "Gender Detection",
            "Speech Recognition", "Named Entity Recognition", "Gesture Recognition", "Person Identification", "Person Recognition",
            "Few Shot", "Few Shot Learning", "Few Shot Classification", "Few Shot Recognition", "Few Shot Detection", "Domain Adaptation",
            "Huggingface", "Knowledge Graph", "Edge Detection", "Focal Loss", "Lossless Compression", "Lossy Compression", "Data Compression",
            "Compression", "Foundation Model", "Segmentation Foundation Model", "ChatGPT Chrome Extention", "GPT", "GPT Chat",
            "Privacy Preserving Machine Learning", "Federated Learning", "machinelearning", "deeplearning", "Multitask Learning",
            "Out Of Distribution", "Out Of Distribution Generalization", "Out Of Distribution Robustness", "Open Set Recognition",
            "OOD Generalization", "OOD Robustness", "OOD Detection", "Open World Learning", "Outlier Detectionf", "Distribution Shifts",
            "generative adversarial network", "GAN", "image interpolation", "image manipulation", "adversarial detection",
            "adversarial samples", "adversarial attack", "action segmentation", "imagesegmentation", "Text Segmentation", "Chinese Text Segmentation",
            "gaussian mixture models", "text to image", "text to diagram", "text to video", "text to image diffusion", "text to url",
            "3d generation", "3d shape generation", "image to sound", "image2text", "text2image", "text2video", "monte carlo", "markov chain",
            "markov chain monte carlo", "hamiltonian monte carlo", "hybrid monte carlo", "inference algorithm", "markov text", "markov namegen",
            "markov decision processes", "kmeans", "kmeans clustering", "clustering algorithm", "kernel kmeans clustering", "kernel method",
            "random forest", "blackbox testing", "regression testing", "visual regression", "image regression", "speech synthesis",
            "Question Answering", "visual question answering", "VQA"]


# Load pre-trained Sentence-BERT model
print("loading model...\n")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentenceTransformer("all-MiniLM-L6-v2")
model = model.to(device)


# Encode keywords into embeddings
keywords = [keyword.lower() for keyword in keywords]
keyword_embeddings = model.encode(keywords, normalize_embeddings=True)



# Function to check if any topic word is similar to the keywords
def is_similar(topics_str, keyword_embeddings, threshold=0.7):
    try:
        topics_list = ast.literal_eval(topics_str)  # Convert string to list
        if not isinstance(topics_list, list) or len(topics_list) == 0:  # Ignore empty lists
            return False
    except:
        return False  # If parsing fails, treat as no match

    # Clean topics: replace special characters with spaces
    cleaned_topics = [re.sub(r'[^a-zA-Z0-9 ]', ' ', topic) for topic in topics_list]

    # Encode each cleaned topic word
    topic_embeddings = model.encode(cleaned_topics, normalize_embeddings=True)

    # Compute cosine similarity between each topic word and keywords
    similarity_matrix = cosine_similarity(topic_embeddings, keyword_embeddings)

    # Check if any word in topics has similarity > threshold
    return (similarity_matrix.max(axis=1) > threshold).any()



# Remove rows with empty topics
print("Original data size: {}".format(len(df)))
df = df[df['topics'].apply(lambda x: len(ast.literal_eval(x)) > 0 if x.strip().startswith('[') else False)]
print("Filtered data size: {}\n".format(len(df)))



# Apply the function to filter rows
df["is_matched"] = df["topics"].apply(lambda x: is_similar(str(x), keyword_embeddings))



# Create two separate dataframes
matched_df = df[df["is_matched"]].drop(columns=["is_matched"])  # Rows where similarity > threshold
unmatched_df = df[~df["is_matched"]].drop(columns=["is_matched"])  # Rows where no match found


# Save matched topics to csv
all_matched_topics = []

for topics in matched_df['topics']:
    topic_list = ast.literal_eval(topics)  # Convert string to list
    all_matched_topics.extend(topic_list)  # Collect all topics

topics_df = pd.DataFrame({'topic': all_matched_topics})
topics_df.to_csv('matched_topics.csv', index=False)

# Save unmatched topics to csv
all_unmatched_topics = []

for topics in unmatched_df['topics']:
    topic_list = ast.literal_eval(topics)  # Convert string to list
    all_unmatched_topics.extend(topic_list)  # Collect all topics

topics_df = pd.DataFrame({'topic': all_unmatched_topics})
topics_df.to_csv('unmatched_topics.csv', index=False)


# Display results
print("Matched Rows:")
print(matched_df['topics'])
print("\n")

print("\nUnmatched Rows:")
print(unmatched_df['topics'])
print("\n")

# Optionally save the results
matched_df.to_csv("AI_repos.csv", index=False)
unmatched_df.to_csv("nonAI_repos.csv", index=False)


matched_df = pd.read_csv("AI_repos.csv")
unmatched_df = pd.read_csv("nonAI_repos.csv")

print("unmatched instances: {}  stars: {}  popularity score 1: {}  popularity score 2: {}".format(len(unmatched_df), unmatched_df["stars"].mean(), unmatched_df["popularity_score_1"].mean(), unmatched_df["popularity_score_2"].mean()))
print("matched instances: {}  stars: {}  popularity score 1: {}  popularity score 2: {}".format(len(matched_df), matched_df["stars"].mean(), matched_df["popularity_score_1"].mean(), matched_df["popularity_score_2"].mean()))
