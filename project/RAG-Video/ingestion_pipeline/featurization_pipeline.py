import base64
import io
import os
import uuid
from typing import Dict, List

import imagehash
import spacy
import torch
from chunking_pipeline import prepare_video_for_embedding
from dotenv import load_dotenv
from PIL import Image
from pymongo import MongoClient, errors
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from transformers import CLIPModel, CLIPProcessor

load_dotenv()
nlp = spacy.load("en_core_web_sm")
MONGO_COLLECTIONS = "decoded_frames"
MONGO_DB_STRING = os.getenv("DATABASE_HOST")
qdrant_url = os.getenv("QDRANT_HOST")
qdrant_key = os.getenv("QDRANT_KEY")
vector_size = 1024
COLLECTION_NAME = "video_chunks"

print("Loading QDRANT")
try:
    client = MongoClient(MONGO_DB_STRING)
    db = client["dataCollection"]
    collection = db[MONGO_COLLECTIONS]
    print(collection.find_one())
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    if not qdrant_client.collection_exists(collection_name="video_chunks"):
        qdrant_client.create_collection(
            collection_name="video_chunks",
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    print("QDRANT LOADED")
except Exception as e:
    print(f"Error: {e}")
except errors.ServerSelectionTimeoutError as e:
    print(f"Error: {e}")


print("Loading Clip Model")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def decode_base64_image(base64_str: str) -> Image.Image:
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def generate_combined_embedding(image: Image.Image, text: str) -> List[float]:
    # Separate encoding for text
    text_inputs = clip_processor(
        text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77
    )
    image_inputs = clip_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        text_embedding = clip_model.get_text_features(**text_inputs)[0].numpy().tolist()
        image_embedding = (
            clip_model.get_image_features(**image_inputs)[0].numpy().tolist()
        )
    return text_embedding + image_embedding


def merge_subtitles(subtitles: List[Dict], time_gap_threshold=1.0) -> List[str]:
    merged_blocks = []
    current_block = []
    prev_end = 0.0

    for entry in subtitles:
        start = float(entry["timestamp"])
        text = entry["subtitle"]
        end = start  # approximate for simple captions with single timestamps

        # Start a new block if time gap exceeds threshold
        if current_block and (start - prev_end) > time_gap_threshold:
            merged_blocks.append(" ".join(current_block))
            current_block = []

        current_block.append(text)
        prev_end = end

    # Final block
    if current_block:
        merged_blocks.append(" ".join(current_block))

    return merged_blocks


def segment_into_sentences(blocks: List[str]) -> List[str]:
    all_sentences = []
    for block in blocks:
        doc = nlp(block)
        all_sentences.extend(
            [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        )
    return all_sentences


def chunk_sentences(sentences: List[str], max_tokens=1000) -> List[Dict]:
    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        token_count = len(nlp(sentence))
        if current_token_count + token_count > max_tokens:
            chunks.append({"text": " ".join(current_chunk)})
            current_chunk = []
            current_token_count = 0
        current_chunk.append(sentence)
        current_token_count += token_count

    if current_chunk:
        chunks.append({"text": " ".join(current_chunk)})

    return chunks


def hash_image(base64_image: str):
    try:
        image = decode_base64_image(base64_image)
        return str(imagehash.phash(image))  # perceptual hash
    except Exception as e:
        print(f"Image hash error: {e}")
        return None


def prepare_video_for_embedding(video_id: str) -> List[Dict]:
    entries = list(collection.find({"video_id": video_id}))
    if not entries:
        print(f"No entries found for video_id: {video_id}")
        return []

    # Sort entries by timestamp
    sorted_entries = sorted(entries, key=lambda x: float(x["timestamp"]))

    # Merge subtitles based on time gap
    merged = merge_subtitles(sorted_entries)
    sentences = segment_into_sentences(merged)
    text_chunks = chunk_sentences(sentences)

    result_chunks = []
    used_hashes = set()

    for i, chunk in enumerate(text_chunks):
        if i < len(sorted_entries):
            frame_entry = sorted_entries[i]
            image_hash = hash_image(frame_entry["frame"])

            if image_hash:
                if image_hash not in used_hashes:
                    # If the image hash is not used, create a new chunk
                    used_hashes.add(image_hash)
                    result_chunks.append(
                        {
                            "text": chunk["text"],
                            "frame": frame_entry["frame"],
                            "video_id": frame_entry["video_id"],
                            "start": frame_entry["timestamp"],
                            "end": frame_entry["timestamp"],
                        }
                    )
                else:
                    # If the hash is already used, modify the last chunk's end time
                    print(f"Similar frame found, extending last chunk at index {i}...")
                    last_chunk = result_chunks[-1]
                    last_chunk["end"] = frame_entry["timestamp"]
            else:
                print(f"Invalid frame at index {i}, skipping...")
        else:
            print(f"Warning: No frame for chunk index {i} in video {video_id}")

    return result_chunks


def featurize_and_store(chunks: List[Dict]):
    points = []
    for chunk in chunks:
        try:
            image = decode_base64_image(chunk["frame"])
            text = chunk["text"]
            embedding = generate_combined_embedding(image, text)

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "video_id": chunk["video_id"],
                    "start": chunk.get("start"),
                    "end": chunk.get("end"),
                    "text": text,
                },
            )
            points.append(point)
        except Exception as e:
            print(f"Skipping chunk due to error: {e}")
    if points:
        print(f"Inserting {len(points)} vectors into Qdrant...")
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)


def chunk_and_embedd_to_qdrant():
    video_ids = collection.distinct("video_id")
    for video_id in video_ids:
        print(f"Processing: {video_id}")
        chunks = prepare_video_for_embedding(video_id)
        if chunks:
            featurize_and_store(chunks)
        else:
            print(f"No chunks in {video_id}")
