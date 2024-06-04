from fastapi import FastAPI, Depends, HTTPException, Header
import os
import sqlite3
import replicate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Set your Replicate API token here
#REPLICATE_API_TOKEN = "r8_QwutdkpdAMmx15JNgjtQvVcMLZnnK9A2tiP8M"
#remove the comments that contain API keys
#api_key = 'sk-proj-8G3aBiWkMJMPca19nh4xT3BlbkFJ67ZOx5FZHRzqY8bbsLCH'
#client = OpenAI(api_key=api_key)


#os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# Function to initialize the database
def initialize_database():
    conn = sqlite3.connect('database.db', check_same_thread=False)
    c = conn.cursor()
    # Create table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS my_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            data TEXT NOT NULL
        )
    ''')
    # Insert sample data if table is empty
    c.execute('SELECT COUNT(*) FROM my_table')
    if c.fetchone()[0] == 0:
        c.execute('''
            INSERT INTO my_table (name, data)
            VALUES
            ('Alice', 'Hello, this is a sample data from Alice.'),
            ('Bob', 'This is another sample data from Bob.'),
            ('Charlie', 'Yet another sample data from Charlie.')
        ''')
    conn.commit()
    conn.close()


# Initialize the database
initialize_database()

# Connect to the database with check_same_thread set to False
conn = sqlite3.connect('database.db', check_same_thread=False)
c = conn.cursor()

app = FastAPI()


# Enable CORS for all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You might want to restrict this to specific origins in production
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
        print("Received a request.")
        message = msg.message
        print("Received message:", message)

        # Query the database
        query = f"SELECT * FROM my_table WHERE data LIKE '%{message}%'"
        print("Executing query:", query)
        c.execute(query)
        rows = c.fetchall()
        print("Rows fetched from database:", rows)

        # Process the database results
        relative_data = [row[2] for row in rows]
        print("Relative data:", relative_data)

        # Get responses from the models
        output = []

        # ChatGPT Turbo Result
        completion_3_5_turbo = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
                {"role": "user", "content": message}
            ]
        )
        gpt_3_5_turbo_response = completion_3_5_turbo.choices[0].message.content
        print("GPT-3.5-turbo response:", gpt_3_5_turbo_response)

        # GPT 4 Result
        completion_gpt_4 = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
                {"role": "user", "content": message}
            ]
        )
        gpt_4_response = completion_gpt_4.choices[0].message.content
        print("GPT-4 response:", gpt_4_response)

        out_llama = replicate.run(
            'meta/llama-2-70b-chat',
            input={"prompt": message}
        )

        output.append(out_llama)
        llama = ''.join(output[0])

        out_falcon = replicate.run(
            'joehoover/falcon-40b-instruct:7d58d6bddc53c23fa451c403b2b5373b1e0fa094e4e0d1b98c3d02931aa07173',
            input={"prompt": message}
        )
        output.append(out_falcon)
        falcon = ''.join(output[1])
        print("Falcon response:", falcon)

        # Calculate cosine similarities
        similarities = []
        for i, data in enumerate(relative_data):
            tfidf_matrix_llama = vectorizer.fit_transform([data, llama])
            cos_sim_llama = cosine_similarity(tfidf_matrix_llama[0:1], tfidf_matrix_llama[1:2])
            similarities.append((i, 'Llama', cos_sim_llama[0][0] * 100))

            tfidf_matrix_falcon = vectorizer.fit_transform([data, falcon])
            cos_sim_falcon = cosine_similarity(tfidf_matrix_falcon[0:1], tfidf_matrix_falcon[1:2])
            similarities.append((i, 'Falcon', cos_sim_falcon[0][0] * 100))
            tfidf_matrix_3_5_turbo = vectorizer.fit_transform([data, gpt_3_5_turbo_response])
            cos_sim_3_5_turbo = cosine_similarity(tfidf_matrix_3_5_turbo[0:1], tfidf_matrix_3_5_turbo[1:2])
            similarities.append((i, 'GPT-3.5-turbo', cos_sim_3_5_turbo[0][0] * 100))

            tfidf_matrix_gpt_4 = vectorizer.fit_transform([data, gpt_4_response])
            cos_sim_gpt_4 = cosine_similarity(tfidf_matrix_gpt_4[0:1], tfidf_matrix_gpt_4[1:2])
            similarities.append((i, 'GPT-4', cos_sim_gpt_4[0][0] * 100))

        # Find the response with the highest cosine similarity
        best_response = max(similarities, key=lambda x: x[2])
        best_index, best_model, best_similarity = best_response
        print("Best Response:", best_model)

        if best_model == 'Llama':
            best_response_text = llama
        elif best_model == 'Falcon':
            best_response_text = falcon
        elif best_model == 'GPT-3.5-turbo':
            best_response_text = gpt_3_5_turbo_response
        else:
            best_response_text = gpt_4_response

        print("Best Response Text:", best_response_text)

        return {'response': best_response_text}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
