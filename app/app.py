from flask import Flask, render_template, request, session,redirect
import google.generativeai as genai
import faiss
import os
import json
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for sessions

# Configure the API key
genai.configure(api_key='YOUR_API_KEY')

# Load the FAISS index and documents mapping
save_folder = "rag_system"
faiss_index_path = os.path.join(save_folder, "index.faiss")
if os.path.exists(faiss_index_path):
    faiss_index = faiss.read_index(faiss_index_path)
else:
    print(f"Index file not found at {faiss_index_path}")
    # Handle the case where the index file is missing
    # Optionally, generate the index here or raise an error

with open(os.path.join(save_folder, "documents.json"), "r") as f:
    docs_mapping = json.load(f)

law_names = [law['law'] for law in docs_mapping.values()]

def embed_query(text):
    embedding_result = genai.embed_content(
        model='models/embedding-001',
        content=text,
        task_type='retrieval_query'
    )
    return np.array(embedding_result['embedding'], dtype=np.float32).reshape(1, -1)

def retriever(query, index, law_names, k=1):
    query_embedding = embed_query(query)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k)
    retrieved_laws = [law_names[indices[0][i]] for i in range(k)]
    return retrieved_laws

def create_prompt(query, retrieved_laws, history=None):
    prompt = "Answer the following question based on the provided laws:\n\n"
    if history:
        for msg in history:
            prompt += f"**{msg['role'].capitalize()}:** {msg['content']}\n\n"
    prompt += f"**Question:** {query}\n\n"
    prompt += "**Relevant Laws:**\n"
    for i, law in enumerate(retrieved_laws, 1):
        prompt += f"{i}. {law}\n"
    prompt += "\nPlease provide a brief answer using the relevant laws."
    return prompt

def generate_response(prompt):
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_query = request.form['query']
        mode = request.form['mode']
        
        # Retrieve relevant laws
        retrieved_laws = retriever(user_query, faiss_index, law_names, k=3)
        
        # Manage conversation history based on mode
        if mode == 'with_memory':
            if 'history' not in session:
                session['history'] = []
            session['history'].append({'role': 'user', 'content': user_query})
            # Create prompt with history
            prompt = create_prompt(user_query, retrieved_laws, session['history'])
            response_text = generate_response(prompt)
            session['history'].append({'role': 'assistant', 'content': response_text})
        else:
            # Without memory, no history
            prompt = create_prompt(user_query, retrieved_laws)
            response_text = generate_response(prompt)
            session.pop('history', None)
        
        return render_template('index.html', response=response_text, retrieved_laws=retrieved_laws, mode=mode)
    else:
        mode = request.args.get('mode', 'with_memory')
        return render_template('index.html', mode=mode)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session.pop('history', None)
    return redirect('/?mode=with_memory')

if __name__ == '__main__':
    app.run(debug=True)