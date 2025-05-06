import os
import sys

import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "retrieval_pipeline"))
)
from retrieval_pipeline import retrieval_pipeline_url

load_dotenv()
API_KEY = os.getenv("HF_TOKEN")
client = InferenceClient(
    provider="novita",
    api_key=API_KEY,
)


# -- LLM Call --
def query_llm(user_query, youtube_link, max_tokens=250):
    prompt = f"""The user is asking a question about a YouTube video.

Video Link: {youtube_link}
Question: {user_query}

Please help the user by answering their question based on the linked video. You may assume the video link points directly to the relevant part of the content."""

    try:
        # Call the Hugging Face API using the InferenceClient
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-Prover-V2-671B",  # replace with the appropriate model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        # Extract the response text from the API response
        return completion.choices[0].message["content"]
    except Exception as e:
        return f"Error communicating with Hugging Face API: {e}"


# -- Main Function --
def handle_query(query):
    url = retrieval_pipeline_url(query)
    answer = query_llm(query, url)

    iframe_html = f'<iframe width="560" height="315" src="{url}" frameborder="0" allowfullscreen></iframe>'
    return answer, iframe_html


# -- Gradio UI --
demo = gr.Interface(
    fn=handle_query,
    inputs=[gr.Textbox(label="Your Question")],
    outputs=[gr.Textbox(label="LLM Answer"), gr.HTML(label="Embedded YouTube Clip")],
    title="Video Q&A with YouTube + LLM",
    description="Ask questions about a YouTube clip. The LLM will answer based on the linked video.",
)

if __name__ == "__main__":
    demo.launch()
