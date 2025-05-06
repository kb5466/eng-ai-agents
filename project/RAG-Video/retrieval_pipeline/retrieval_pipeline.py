import base64
import os
from difflib import SequenceMatcher
from io import BytesIO

import numpy as np
import qdrant_client
import spacy
import torch
from bertopic import BERTopic
from dotenv import load_dotenv
from PIL import Image
from pymongo import MongoClient, errors
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from transformers import BlipModel, BlipProcessor

sentence_model = SentenceTransformer("all-MiniLM-L12-v2")

nlp = spacy.load("en_core_web_sm")
collection_name = "video_chunks"

load_dotenv()
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_KEY = os.getenv("QDRANT_KEY")
MONGO_COLLECTIONS = "decoded_frames"
MONGO_DB_STRING = os.getenv("DATABASE_HOST")

# Connecting to Database
try:
    MongoClient = MongoClient(MONGO_DB_STRING)
    qdrant_client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_KEY)
    db = MongoClient["dataCollection"]
    collection = db[MONGO_COLLECTIONS]
    print("Database Ready.")
except Exception as e:
    print(f"Error: {e}")
except errors.ServerSelectionTimeoutError as e:
    print(f"Error: {e}")

print("Loading fine-tuned model")

MODEL_PATH = "../Ingestion-Pipeline/blip-finetuned"
model = BlipModel.from_pretrained(MODEL_PATH)
processor = BlipProcessor.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Loading BERTopic()")
topic_model = BERTopic()


def decode_base64_to_image(base64_str):
    """Decode base64 string to PIL RGB image."""
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data)).convert("RGB")


def get_query_embedding(query: str):
    inputs = processor(text=query, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        embeddings = model.get_text_features(**inputs)
    return embeddings.cpu().numpy().flatten()


def get_sentence_embeddings(texts):
    return np.array([get_query_embedding(text) for text in texts])


def is_similar(a, b, threshold=0.85):
    return SequenceMatcher(None, a, b).ratio() > threshold


def get_image_embedding(image_data: str):
    """
    Converts an image (base64 string) to embeddings using the fine-tuned BLIP model.
    """
    image = decode_base64_to_image(image_data)
    inputs = processor(images=image, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)

    return embeddings.cpu().numpy().flatten()


def deduplicate(sentences):
    unique = []
    for sent in sentences:
        if all(not is_similar(sent, seen) for seen in unique):
            unique.append(sent)
    return unique


def store_embeddings_in_qdrant(texts, embeddings, timestamps, video_ids):
    if not qdrant_client.collection_exists("video_chunks"):
        qdrant_client.create_collection(
            "video_chunks",
            vector_params=VectorParams(size=embeddings.shape[1], distance="Cosine"),
        )

    points = [
        PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload={"text": text, "timestamp": timestamp, "video_id": vid},
        )
        for idx, (text, embedding, timestamp, vid) in enumerate(
            zip(texts, embeddings, timestamps, video_ids, strict=False)
        )
    ]
    qdrant_client.upsert(collection_name="video_chunks", points=points)


def process_video_sentences(video_id):
    docs = collection.find({"video_id": video_id})
    full_text, timestamp_map = "", []

    for doc in docs:
        full_text += " " + doc["subtitle"]
        timestamp_map.append(doc["timestamp"])

    doc = nlp(full_text.strip())
    sentences = [sent.text.strip() for sent in doc.sents]

    # Simplistic timestamp assignment (can be improved)
    n = len(sentences)
    if n == 0:
        return [], [], []
    timestamps = (
        np.linspace(min(timestamp_map), max(timestamp_map), n).astype(int).tolist()
    )
    return sentences, timestamps, [video_id] * n


def resize_embedding(embedding, target_dim=512):
    """
    Resizes the embedding to the target dimension, padding with zeros if necessary.
    """
    current_dim = len(embedding)
    if current_dim < target_dim:
        # Zero padding if the current dimension is smaller than target
        return np.pad(embedding, (0, target_dim - current_dim), mode="constant")
    elif current_dim > target_dim:
        # Truncate if the current dimension is larger than target
        return embedding[:target_dim]
    return embedding


def search_qdrant(query_embedding, top_k=5):
    # Perform the query and get the results
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding.tolist(),  # The embedding query vector
        limit=top_k,  # Number of results
        with_payload=True,  # Include payload (video_id, start, end, etc.)
        with_vectors=False,  # Exclude vectors if not needed
    )

    # Results should contain the actual points with their scores
    # Results returned from `query_points` might be a dict where 'result' contains the matching points
    extracted_results = []
    for point in results.points:
        video_id = point.payload.get("video_id")
        start = point.payload.get("start")
        end = point.payload.get("end")
        text = point.payload.get("text")
        frame = point.payload.get("frame")

        # Check if the video_id and timestamp (start) already exist in the extracted_results to avoid duplicates
        if not any(
            result["video_id"] == video_id and result["start"] == start
            for result in extracted_results
        ):
            extracted_results.append(
                {
                    "video_id": video_id,
                    "start": start,
                    "end": end,
                    "text": text,
                    "frame": frame,
                }
            )
            print(f"Video ID: {video_id}, Timestamp: {start} - {end}, Text: {text}")

    return extracted_results


def retrieval_pipeline(search_query: str, top_k: int = 5):
    """
    Accepts a user query, performs a semantic search using the fine-tuned BLIP model,
    compares it to the retrieved results using Sentence-Transformer, and generates
    a video clip from the top matching result.
    """
    print(f"\n🔍 Searching for: {search_query}")

    # Step 1: Embed the user query using the sentence model (SentenceTransformer)
    query_embedding = sentence_model.encode([search_query])[0]

    # Resize the query embedding to 512 dimensions
    query_embedding = resize_embedding(query_embedding)

    # Step 2: Search in Qdrant
    results = search_qdrant(
        query_embedding, top_k=top_k
    )  # [{"video_id": video_id, "start": start, "end": end, "text": text, "frame": frame}...]

    if not results:
        print("No relevant results found.")
        return

    # Step 3: Collect the texts from the results and embed them
    texts = [r["text"] for r in results]
    text_embeddings = sentence_model.encode(texts)

    # Resize the text embeddings to 512 dimensions
    text_embeddings_resized = np.array([resize_embedding(te) for te in text_embeddings])

    # Step 4: Calculate cosine similarity between the query embedding and the result embeddings
    similarities = []
    for idx, result in enumerate(results):
        similarity = 1 - cosine(query_embedding, text_embeddings_resized[idx])
        similarities.append((result, similarity))

    # Step 5: Sort results based on similarity (descending order)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Step 6: Pick the top result (highest similarity)
    top_result = similarities[0][0]
    best_video_id = top_result.get("video_id")

    # Step 7: Find all results for the same video_id (to handle multiple segments within the same video)
    related_results = [
        result for result in results if result["video_id"] == best_video_id
    ]

    # Step 8: Select the start and end timestamps for the clip (adding a 5s padding)
    start = min([r["start"] for r in related_results])
    end = max([r["end"] for r in related_results]) + 5
