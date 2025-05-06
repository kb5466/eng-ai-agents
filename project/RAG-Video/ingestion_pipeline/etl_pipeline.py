import base64
import gc
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import imagehash
import nltk
import webdataset as wds
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
from PIL import Image
from pymongo import MongoClient, errors

load_dotenv()


nltk.download("punkt")

MONGO_COLLECTIONS = "decoded_frames"
VIDEO_TIME = 60
# Check for database credentials
MONGO_DB_STRING = os.getenv("DATABASE_HOST")

if not MONGO_DB_STRING:
    print("ERROR: MongoDB Connection String Missing")

# Connecting to Database
try:
    client = MongoClient(MONGO_DB_STRING)
    db = client["dataCollection"]
    collection = db["decoded_frames"]
    print(collection.find_one())
except errors.ServerSelectionTimeoutError as e:
    print(f"Error: {e}")


def frame_to_base64(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


### For Streaming Dataset
def load_data_set():
    hf_token = os.getenv("HF_TOKEN")
    repo = "aegean-ai/ai-lectures-spring-24"
    url = f"pipe:curl -s -L https://huggingface.co/datasets/{repo}/resolve/main/youtube_dataset.tar -H 'Authorization: Bearer {hf_token}'"

    try:
        dataset = wds.WebDataset(url).decode().to_tuple("mp4", "json", "__key__")
        print("Data set loaded.")
        return dataset
    except Exception as e:
        print(f"Failed to process {e}")


# Frame Extraction
def extract_frames(video_path, fps=1):
    cap = cv2.VideoCapture(video_path)
    count = 0
    success = True
    while success:
        success, frame = cap.read()
        if not success:
            break
        if count % int(cap.get(cv2.CAP_PROP_FPS) // fps) == 0:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield frame, timestamp
        count += 1
    cap.release()


# Parse and Align Subtitles
def align_frame_with_subtitles(
    timestamp, subtitles, tolerance=0.5
):  # Array of json [{start, end, text}, ...]
    for entry in subtitles:
        start = timestamp_to_seconds(entry["start"])
        end = timestamp_to_seconds(entry["end"])
        if start - tolerance <= timestamp <= end + tolerance:
            return entry["text"]
    return None


# Converting Time
def timestamp_to_seconds(time_str):
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


# Remove Redundant Frames
def redundant_frames(new_frame, old_frame, threshold=5):
    hash1 = imagehash.phash(Image.fromarray(new_frame))
    hash2 = imagehash.phash(Image.fromarray(old_frame))
    return abs(hash1 - hash2) < threshold


def process_captions(caption_entries):
    all_sentences = []
    for entry in caption_entries:
        sentences = sent_tokenize(entry["text"])
        for s in sentences:
            all_sentences.append(
                {"start": entry["start"], "end": entry["end"], "sentence": s}
            )
    return all_sentences


def stream_and_process_dataset(dataset, max_workers=2):
    def process_entry(video, metadata, key):
        video_id = metadata.get("video_id", key)
        title = metadata.get("title", "Untitled")
        subtitles = metadata.get("captions", [])
        ### Decodes frames
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp_video:
                print(f"Processing: {title}")
                tmp_video.write(video)
                tmp_video.flush()

                previous_frame = None
                for frame, timestamp in extract_frames(tmp_video.name):
                    if previous_frame is not None and redundant_frames(
                        frame, previous_frame
                    ):
                        continue

                    subtitle_text = align_frame_with_subtitles(timestamp, subtitles)
                    if not subtitle_text:
                        continue

                    encoded_frame = frame_to_base64(frame)

                    # Insert into MongoDB
                    collection.insert_one(
                        {
                            "video_id": video_id,
                            "timestamp": timestamp,
                            "subtitle": subtitle_text,
                            "frame": encoded_frame,
                        }
                    )
                    previous_frame = frame
                del video, metadata, subtitles, previous_frame
                gc.collect()
                print(f"{title} Done.")
        except Exception as e:
            print(f"Error: {e}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for video, metadata, key in dataset:
            executor.submit(process_entry, video, metadata, key)


dataset = load_data_set()
stream_and_process_dataset(dataset)
