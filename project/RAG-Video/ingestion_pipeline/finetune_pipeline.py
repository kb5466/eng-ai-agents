import base64
import os
import random
from io import BytesIO

import pandas as pd
import torch
from dotenv import load_dotenv
from PIL import Image
from pymongo import MongoClient, errors
from torch.utils.data import Dataset
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    Trainer,
    TrainingArguments,
)

load_dotenv()
MONGO_COLLECTIONS = "decoded_frames"
VIDEO_TIME = 60
MONGO_DB_STRING = os.getenv("DATABASE_HOST")

if not MONGO_DB_STRING:
    print("ERROR: MongoDB Connection String Missing")

# Connecting to Database
try:
    client = MongoClient(MONGO_DB_STRING)
    db = client["dataCollection"]
    collection = db[MONGO_COLLECTIONS]
    print("Database Ready.")
except errors.ServerSelectionTimeoutError as e:
    print(f"Error: {e}")

cursor = collection.find(
    {"frame": {"$exists": True}, "subtitle": {"$exists": True}},
    {"frame": 1, "subtitle": 1},
).limit(1000)
docs = list(cursor)
sampled_docs = random.sample(docs, 500)  # sample 500 randomly in Python
data = pd.DataFrame(sampled_docs).dropna(subset=["frame", "subtitle"])

# Rename columns
data = data[["frame", "subtitle"]].rename(columns={"subtitle": "text"})


class MongoImageCaptionDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.data = dataframe.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def decode_base64_to_image(self, base64_str):
        """Decode base64 string to PIL RGB image."""
        image_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(image_data)).convert("RGB")

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = self.decode_base64_to_image(row["frame"])
        text = row["text"]
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": inputs["input_ids"].squeeze(0),  # <-- This is critical
        }


# Initialize processor, model, dataset
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model is on: {model.device}")

dataset = MongoImageCaptionDataset(data, processor)

# Set up training args
output_dir = os.path.join(os.path.abspath("."), "blip-finetuned")
training_args = TrainingArguments(
    output_dir="./blip-finetuned",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    eval_strategy="no",
    remove_unused_columns=False,
    fp16=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# Save model and processor
print("Saving model and processor...")
print(f"Saving model to {output_dir}")
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print("Model and processor saved.")
