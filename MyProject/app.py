import os
import gradio as gr
from pypdf import PdfReader
import google.generativeai as genai

# Setup Gemini API (Reads key from environment variables)
if "GEMINI_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def ask_gemini_about_docs(file_objects, question):
    # Safety checks
    if not file_objects:
        return "⚠️ Please upload at least one .txt or .pdf file."
    if not question.strip():
        return "⚠️ Please type a valid question."
        
    # Check if the API key is actually set before making the call
    if "GEMINI_API_KEY" not in os.environ or not os.environ["GEMINI_API_KEY"]:
        return "❌ Error: GEMINI_API_KEY environment variable is not set. Please read the testing instructions."

    combined_context = ""
    
    # Extract and join text from all uploaded documents on the fly
    for file_obj in file_objects:
        # file_obj can be a string path or a Gradio file object wrapper depending on version
        file_path = file_obj.name if hasattr(file_obj, "name") else file_obj
        base_name = os.path.basename(file_path)
        
        try:
            if base_name.lower().endswith(".txt"):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    combined_context += f"\n--- Start of document: {base_name} ---\n"
                    combined_context += f.read() + "\n"
            elif base_name.lower().endswith(".pdf"):
                reader = PdfReader(file_path)
                combined_context += f"\n--- Start of document: {base_name} ---\n"
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        combined_context += text + "\n"
        except Exception as e:
            return f"❌ Error parsing file {base_name}: {e}"

    # Query the cloud-hosted Gemini model
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""You are a helpful assistant. Use the following pieces of context to answer the user's question accurately. 
If you don't know the answer based on the text context provided, explicitly say that you don't know. Keep the answer concise.
        
Document Context:
{combined_context}

User Question: {question}
Answer:"""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as err:
        return f"🚨 API Connection Error: {err}"

# --- Build Gradio Interface Layer ---
with gr.Blocks(title="Cloud Multi-Doc RAG") as demo:
    gr.Markdown("# ☁️ Free Cloud RAG Document Assistant")
    gr.Markdown("Test your app locally using Gemini 1.5 Flash before pushing it live to Render.")
    
    with gr.Row():
        file_uploader = gr.File(label="Upload documents (.txt or .pdf)", file_count="multiple", file_types=[".txt", ".pdf"])
        
    with gr.Row():
        question_input = gr.Textbox(label="Ask a question about your files:", placeholder="e.g., What are the terms of the agreement?")
        
    ask_btn = gr.Button("🚀 Analyze Files & Answer", variant="primary")
    answer_output = gr.Textbox(label="AI Response:", interactive=False, lines=10)

    ask_btn.click(fn=ask_gemini_about_docs, inputs=[file_uploader, question_input], outputs=[answer_output])

# Render web port configuration defaults to 10000
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)