# Video RAG Pipelines

This project contains a modular RAG (Retrieval-Augmented Generation) system for educational video content, split into two main pipelines:

* **Ingestion Pipeline**: Extracts, processes, and embeds video frames + subtitles.
* **Retrieval Pipeline**: Allows natural language search over the processed content and generates relevant clips.

---

## Ingestion Pipeline

### `etl_pipeline.py`

* Extracts raw video and subtitle data.
* Converts video into frames and associates each frame with the nearest subtitle segment.
* Stores frame-subtitle pairs into MongoDB (`decoded_frames` collection).

### `chunking_pipeline.py`

Prepares raw subtitle and frame data for embedding by breaking down long transcripts into manageable chunks aligned with visual context.

#### Steps:

1. **Fetch Entries** from MongoDB for a given `video_id`, sorted by timestamp.
2. **Merge Subtitles** into larger blocks based on temporal gaps (default: >1s).
3. **Sentence Segmentation** using spaCy to ensure grammatical integrity.
4. **Chunking**: Sentences are grouped into chunks with a maximum token count (default: 1000).
5. **Frame Deduplication**: Uses perceptual hashing (`imagehash.phash`) to identify and remove visually duplicate frames.
6. **Output Format**:

   ```python
   {
       "video_id": ...,
       "start": ...,
       "end": ...,
       "text": ...,
       "frame": ...
   }
   ```

---

### `featurization_pipeline.py`

* Loads the processed frame-subtitle chunks from MongoDB.
* Uses OpenAI CLIP to generate text and image embeddings.
* Stores merged embeddings and metadata into Qdrant for semantic retrieval.

---

### `finetune_pipeline.py`

* Fine-tunes BLIP (Bootstrapped Language Image Pretraining) model on your subtitle-frame dataset.
* Improves alignment between video content and text queries.

---

## Retrieval Pipeline

### `retrieval_pipeline.py`

* Accepts a user query.
* Embeds the query using Sentence-Transformer.
* Searches Qdrant for the top-k most semantically similar chunks.
* Re-ranks results using cosine similarity.
* Selects the most relevant clip based on combined frame/text match.
* Uses ffmpeg to extract and return a highlight video segment from the original source.

---

## Project Structure

```
RAG-Project/
│
├── Ingestion-Pipeline/
│   ├── blip-finetuned/            # Fine-tuned BLIP weights
│   ├── chunkin
```
