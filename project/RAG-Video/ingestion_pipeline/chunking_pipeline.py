import base64
import datetime
import io
import os
from typing import Dict, List

import imagehash
import spacy
from dotenv import load_dotenv
from PIL import Image
from pymongo import MongoClient, errors

load_dotenv()


MONGO_COLLECTIONS = "decoded_frames"
MONGO_DB_STRING = os.getenv("DATABASE_HOST")

nlp = spacy.load("en_core_web_sm")

if not MONGO_DB_STRING:
    print("ERROR: MongoDB Connection String Missing")

# Connecting to Database
try:
    client = MongoClient(MONGO_DB_STRING)
    db = client["dataCollection"]
    collection = db[MONGO_COLLECTIONS]
    print(collection.find_one())
except errors.ServerSelectionTimeoutError as e:
    print(f"Error: {e}")


def decode_base64_image(base64_str: str) -> Image.Image:
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def timestamp_to_seconds(time_str):
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def seconds_to_timestamp(seconds):
    return str(datetime.utcfromtimestamp(seconds).strftime("%H:%M:%S.%f")[:-3])


def merge_subtitles(subtitles: List[Dict], time_gap_threshold=1.0) -> List[str]:
    merged_blocks = []
    current_block = []
    prev_end = 0.0

    for entry in subtitles:
        start = float(entry["timestamp"])
        text = entry["subtitle"]
        end = start  # for simplicity

        if current_block and (start - prev_end) > time_gap_threshold:
            merged_blocks.append(" ".join(current_block))
            current_block = []

        current_block.append(text)
        prev_end = end

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


def hash_image(base64_image: str):
    try:
        image = decode_base64_image(base64_image)
        return str(imagehash.phash(image))  # perceptual hash
    except Exception as e:
        print(f"Image hash error: {e}")
        return None


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


def prepare_video_for_embedding(video_id: str) -> List[Dict]:
    entries = list(collection.find({"video_id": video_id}))
    if not entries:
        print(f"No entries found for video_id: {video_id}")
        return []

    sorted_entries = sorted(entries, key=lambda x: float(x["timestamp"]))
    merged = merge_subtitles(sorted_entries)
    sentences = segment_into_sentences(merged)
    text_chunks = chunk_sentences(sentences)

    result_chunks = []
    used_hashes = {}

    for i, chunk in enumerate(text_chunks):
        if i < len(sorted_entries):
            frame_entry = sorted_entries[i]
            timestamp = float(frame_entry["timestamp"])
            image_hash = hash_image(frame_entry["frame"])

            if image_hash:
                if image_hash not in used_hashes:
                    used_hashes[image_hash] = {
                        "text": chunk["text"],
                        "frame": frame_entry["frame"],
                        "video_id": frame_entry["video_id"],
                        "start": timestamp,
                        "end": timestamp,
                    }
                else:
                    # Update the end time and merge text if frame already seen
                    used_hashes[image_hash]["end"] = timestamp
                    used_hashes[image_hash]["text"] += " " + chunk["text"]
            else:
                print(f"Invalid frame at index {i}, skipping...")
        else:
            print(f"Warning: No frame for chunk index {i} in video {video_id}")

    result_chunks = list(used_hashes.values())
    return result_chunks
