import os
import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import replicate

# Set your Replicate API token here
REPLICATE_API_TOKEN = "r8_ALnWEJhnoY9VDhDUUd0wy7WA1vHGX182eDdng"

api_key = 'sk-proj-oQLhy5vdSTtNcQ59wpDHT3BlbkFJk7sKsC1vPtIDj0xgX2Un'
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

client = OpenAI(api_key=api_key)

# Function to initialize the database


conn = sqlite3.connect('database.db', check_same_thread=False)
c = conn.cursor()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorizer = TfidfVectorizer()

class Message(BaseModel):
    message: str

@app.post('/chat')
async def chat(msg: Message):
    try:
        message = msg.message
        print(f"Received message: {message}")

        # Fetch all documents from the database
        c.execute("SELECT * FROM my_table")
        rows = c.fetchall()
        print(f"Fetched rows: {rows}")

        doc_ids = [row[0] for row in rows]
        doc_names = [row[1] for row in rows]
        doc_texts = [row[2] for row in rows]

        # Calculate TF-IDF vectors and cosine similarities
        tfidf_matrix = vectorizer.fit_transform([message] + doc_texts)
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        print(f"Cosine similarities: {cosine_similarities}")

        # Get the top 3 most similar documents
        top_indices = cosine_similarities.argsort()[-1:][::-1]
        top_docs = [(doc_ids[i], doc_names[i], doc_texts[i]) for i in top_indices]
        print(f"Top documents: {top_docs}")

        # Prepare a single string from the top documents
        search_results = "\n\n".join([f"{doc[1]}: {doc[2]}" for doc in top_docs])
        combined_prompt = f"User query: {message}\n\nSearch results:\n{search_results}"
        print(f"Combined prompt: {combined_prompt}")

        # Get responses from the models
        completion_3_5_turbo = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant."},
                {"role": "user", "content": combined_prompt}
            ]
        )
        gpt_3_5_turbo_response = completion_3_5_turbo.choices[0].message.content
        print(f"GPT-3.5-turbo response: {gpt_3_5_turbo_response}")

        completion_gpt_4 = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant."},
                {"role": "user", "content": combined_prompt}
            ]
        )
        gpt_4_response = completion_gpt_4.choices[0].message.content
        print(f"GPT-4 response: {gpt_4_response}")

        out_llama = replicate.run(
            'meta/llama-2-70b-chat',
            input={"prompt": combined_prompt}
        )
        llama_response = ''.join(out_llama)
        print(f"Llama response: {llama_response}")

        out_falcon = replicate.run(
            'joehoover/falcon-40b-instruct:7d58d6bddc53c23fa451c403b2b5373b1e0fa094e4e0d1b98c3d02931aa07173',
            input={"prompt": combined_prompt}
        )
        falcon_response = ''.join(out_falcon)
        print(f"Falcon response: {falcon_response}")

        # Calculate cosine similarities between the responses and the search results
        similarities = []
        responses = {
            'GPT-3.5-turbo': gpt_3_5_turbo_response,
            'GPT-4': gpt_4_response,
            'Llama': llama_response,
            'Falcon': falcon_response
        }

        for model_name, response in responses.items():
            tfidf_matrix_model = vectorizer.fit_transform([search_results, response])
            cos_sim = cosine_similarity(tfidf_matrix_model[0:1], tfidf_matrix_model[1:2])
            similarities.append((model_name, cos_sim[0][0] * 100))
            print(f"Cosine similarity for {model_name}: {cos_sim[0][0] * 100}")

        # Find the response with the highest cosine similarity
        best_model, best_similarity = max(similarities, key=lambda x: x[1])
        best_response_text = responses[best_model]
        print(f"Best model: {best_model} with similarity: {best_similarity}")
        print(f"Best response text: {best_response_text}")

        if best_similarity < 10.0:
            return {'response': "Sorry I don't know!"}


        return {'response': best_response_text}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
