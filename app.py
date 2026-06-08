import os
import gradio as gr
import pypdf
import google.generativeai as genai

def analyze_documents(uploaded_files, user_question, api_key_input):
    # Use the API key provided in the web input field
    if not api_key_input:
        return "❌ Error: Please paste your Gemini API Key into the input field above."
    
    try:
        genai.configure(api_key=api_key_input)
        
        if not uploaded_files:
            return "Please upload at least one document."
            
        combined_text = ""
        for file in uploaded_files:
            if file.name.endswith('.pdf'):
                reader = pypdf.PdfReader(file.name)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        combined_text += text + "\n"
            elif file.name.endswith('.txt'):
                with open(file.name, 'r', encoding='utf-8') as f:
                    combined_text += f.read() + "\n"
        
        if not combined_text.strip():
            return "Could not extract text from the uploaded document(s)."
            
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Context:\n{combined_text}\n\nQuestion: {user_question}")
        return response.text
        
    except Exception as e:
        return f"❌ Error running model: {str(e)}"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Free Cloud RAG Document Assistant")
    gr.Markdown("Paste your API key, drop your documents, and ask away!")
    
    # Secure API Key password field right in the UI
    api_key_input = gr.Textbox(
        label="Enter Gemini API Key (Starts with AIzaSy...)", 
        type="password", 
        placeholder="Paste your key here..."
    )
    
    file_input = gr.File(label="Upload documents (.txt or .pdf)", file_count="multiple")
    question_input = gr.Textbox(label="Ask a question about your files:")
    submit_btn = gr.Button("🚀 Analyze Files & Answer", variant="primary")
    output_text = gr.Textbox(label="AI Response:")
    
    submit_btn.click(
        fn=analyze_documents, 
        inputs=[file_input, question_input, api_key_input], 
        outputs=output_text
    )

demo.launch()
