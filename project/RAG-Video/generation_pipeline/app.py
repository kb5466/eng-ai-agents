import os
import sys

import gradio as gr

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Retrieval-Pipeline"))
)
from retrieval_pipeline import retrieval_pipeline_url


def handle_query(query):
    embed_url = retrieval_pipeline_url(query)
    answer = query_llm(query, embed_url)

    iframe_html = f'<iframe width="560" height="315" src="{embed_url}" frameborder="0" allowfullscreen></iframe>'
    return answer, iframe_html


def retrieval_pipeline_url(search_query: str, top_k: int = 5):
    """
    Accepts a user query, performs a semantic search using the fine-tuned BLIP model,
    compares it to the retrieved results using Sentence-Transformer, and generates
    an embeddable YouTube clip URL from the top matching result.
    """
    print(f"\n Searching for: {search_query}")

    # Step 1: Embed the user query
    query_embedding = sentence_model.encode([search_query])[0]
    query_embedding = resize_embedding(query_embedding)

    # Step 2: Search in Qdrant
    results = search_qdrant(query_embedding, top_k=top_k)
    if not results:
        print("No relevant results found.")
        return None

    # Step 3: Embed result texts
    texts = [r["text"] for r in results]
    text_embeddings = sentence_model.encode(texts)
    text_embeddings_resized = np.array([resize_embedding(te) for te in text_embeddings])

    # Step 4: Compute cosine similarities
    similarities = []
    for idx, result in enumerate(results):
        similarity = 1 - cosine(query_embedding, text_embeddings_resized[idx])
        similarities.append((result, similarity))

    # Step 5: Sort results by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Step 6: Pick top result
    top_result = similarities[0][0]
    best_video_id = top_result.get("video_id")

    # Step 7: Group by video ID
    related_results = [
        result for result in results if result["video_id"] == best_video_id
    ]
    print(best_video_id)

    # Step 8: Calculate clip start/end
    start = int(min([r["start"] for r in related_results]))
    end = int(max([r["end"] for r in related_results])) + 7
    print(f"Timestamp: {start} - {end}")

    # ✅ RETURN FULL EMBED URL (ready for iframe)
    url = f"https://www.youtube.com/embed/{best_video_id}?start={start}&end={end}&autoplay=1"
    return url


with gr.Blocks(title="Video Q&A with YouTube + LLM") as demo:
    gr.Markdown("## Video Q&A with YouTube + LLM")
    gr.Markdown(
        "Ask questions about a YouTube video. The LLM will answer based on the retrieved clip."
    )

    with gr.Row():
        with gr.Column(scale=3):
            query_box = gr.Textbox(
                label="Your Question", placeholder="Ask something about the video..."
            )
            submit_btn = gr.Button("Submit")
        with gr.Column(scale=2):
            video_display = gr.HTML(label="Embedded YouTube Clip")

    llm_answer = gr.Textbox(label="LLM Answer", lines=5)

    submit_btn.click(
        fn=handle_query, inputs=query_box, outputs=[llm_answer, video_display]
    )

if __name__ == "__main__":
    demo.launch()
