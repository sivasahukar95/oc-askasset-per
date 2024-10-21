from milvus_wrapper import MilvusWrapper
from tech_keywords import tech_keywords  # Import tech keywords from the new file
import json
import os
from dotenv import load_dotenv
from ibm_watson_machine_learning.foundation_models import Model
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize MilvusWrapper
db_path = "./milvus_database.db"
collection_name = "asset_collection"
milvus_wrapper = MilvusWrapper(db_path, collection_name)
milvus_wrapper.delete_all_data()
csv_file_path = "./assethub_oct16th2024.csv"
data = milvus_wrapper.load_data(csv_file_path)
milvus_wrapper.insert_data(data)

# Default LLM parameters
DEFAULT_LLM_PARAMS = {
    'decoding_method': "greedy",
    'min_new_tokens': 1,
    'max_new_tokens': 400,
    'repetition_penalty': 1,
    'random_seed': 42,
}

# Models
class QueryRequest(BaseModel):
    user_query_text: str
    limit: int = 3

class SearchResult(BaseModel):
    id: str
    title: str
    author: str
    description: str

DEFAULT_PROMPT = '''
You are a helpful, respectful, and honest assistant. You will be provided with a set of search results and a user query. 
Use the search results to answer the user's query as accurately as possible. Make sure to include the relevant title 
and author name from the search results in your response. However, if the user query already mentions an author, 
do not include that author's name in the response.

Ensure that your answer is clear, concise, and based only on the provided search results. If the query is not related to 
the search results or cannot be answered using the given information, kindly inform the user that the relevant information 
is not available.

User query: {user_query_text}

Search results:
{search_results}

I need ID in the answer wrt the search.
Provide the most relevant answer, including the ID, title, author name and description. 
'''

class LLMBackendCloud:
    def __init__(self, model_id='meta-llama/llama-3-70b-instruct', model_params=DEFAULT_LLM_PARAMS):
        load_dotenv()
        api_key = os.getenv("API_KEY", None)
        ibm_cloud_url = os.getenv("API_ENDPOINT", None)
        project_id = os.getenv("PROJECT_ID", None)
        if api_key is None or ibm_cloud_url is None or project_id is None:
            raise ValueError("Ensure the .env file is in place with correct API details")
        self.creds = {
            "url": ibm_cloud_url,
            "apikey": api_key
        }
        self.model_id = model_id
        self.model_params = model_params
        self.model = Model(model_id=self.model_id, params=self.model_params, credentials=self.creds, project_id=project_id)

    def generate_response(self, prompt: str, model_params=None):
        """Generate a response from the LLM based on the provided prompt."""
        if not model_params:
            model_params = self.model_params
        result = self.model.generate_text(prompt=prompt, params=model_params)
        return result


def extract_author_from_query(query):
    """Extract the author's name from a user query, focusing only on human-like name patterns."""
    match = re.search(r'\b(?:by|authored by|written by|created by|related to|associated with|developed by|connected to|built by|made by|prepared by) ([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b', query, re.IGNORECASE)
    
    if match:
        author_candidate = match.group(1).strip()
        if author_candidate.lower() not in [keyword.lower() for keyword in tech_keywords]:
            return author_candidate
    
    possessive_match = re.search(r'([A-Z][a-z]+(?: [A-Z][a-z]+)+)\'s (?:assets|work)', query, re.IGNORECASE)
    if possessive_match:
        author_candidate = possessive_match.group(1).strip()
        if author_candidate.lower() not in [keyword.lower() for keyword in tech_keywords]:
            return author_candidate

    return None


def extract_entities_from_query(query):
    """Extract important keywords or entities like technologies, tools, or tasks."""
    entities = []
    
    query = re.sub(r'\bwhat\b|\blist\b|\ball\b|\bfor\b|\basset\b|\bassets\b|\brelated to\b|\bshow\b|\bgive\b|\bretrieve\b|\babout\b|\bon\b|\bof\b|\bassociated with\b|\bme\b|\bthe\b|\ba\b|\blinked to\b|\blinked with\b|\bis\b|\bare\b|\bin\b|\bto\b|\band\b|\bwith\b|\bfrom\b', '', query, flags=re.IGNORECASE).strip()
    
    for keyword in tech_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', query, re.IGNORECASE):
            entities.append(keyword)

    if not entities:
        search_terms = re.findall(r'\b([a-zA-Z\s]+)', query)
        for term in search_terms:
            term = term.strip()
            if len(term) > 2: 
                entities.append(term)

    return entities if entities else None

def extract_relevant_keywords(query):
    """Extract relevant keywords including authors and possible entities/technologies."""
    query = query.lower()
    
    author = extract_author_from_query(query)
    entities = extract_entities_from_query(query)

    if author and author.lower() in [e.lower() for e in entities]:
        entities = [e for e in entities if e.lower() != author.lower()]

    return {
        'author': author if author else None,
        'entities': entities
    }

def prepare_prompt(user_query_text, search_results):
    """
    Prepare the final prompt to send to the LLM, ensuring the LLM outputs a JSON-like string format.
    """
    # Prepare the list of assetId dictionaries
    formatted_results = [
        {"assetId": result['id']} for result in search_results
    ]

    # Convert the list of dictionaries to a string in the required format
    results_str = json.dumps(formatted_results)  # Use json.dumps to ensure valid JSON format

    # Return the string prompt with clear instructions
    return f"""
    You are a helpful assistant. The user has requested information based on the following query: "{user_query_text}".

    Please return the following output format:

    {results_str}
    """



def map_hashed_ids_to_original(milvus_wrapper, search_results):
    """Map hashed IDs to original IDs using the stored id_mapping in milvus_wrapper."""
    for result in search_results:
        original_id = milvus_wrapper.id_mapping.get(result['id'], "Unknown")
        result['id'] = original_id  # Replace hashed ID with original ID
    return search_results

def clean_results(results):
    """Remove vector fields and duplicates from the search results."""
    cleaned_results = []
    seen = set()

    for result in results:
        key = (result['title'], result['description'])

        if key not in seen:
            cleaned_result = {
                "id": result.get('id'),
                "title": result.get('title'),
                "author": result.get('author'),
                "description": result.get('description'),
                "type": result.get('type')
            }
            cleaned_results.append(cleaned_result)
            seen.add(key)

    return cleaned_results

def search_by_keywords_or_author(milvus_wrapper, keywords, user_query_text, limit=3):
    author = keywords.get('author')
    entities = keywords.get('entities')

    if author:
        author_search_result = milvus_wrapper.search_by_author(author, limit=limit)
        cleaned_results = json.loads(author_search_result)
    elif entities:
        keyword_search_result = milvus_wrapper.search_by_keywords(entities, limit=limit)
        cleaned_results = json.loads(keyword_search_result)
    else:
        vector_search_result = milvus_wrapper.search(user_query_text, limit=limit)
        cleaned_results = milvus_wrapper.clean_result(vector_search_result)

    cleaned_results = map_hashed_ids_to_original(milvus_wrapper, cleaned_results)

    return cleaned_results



@app.get("/")
async def root():
    return {"message": "Your application is up and running"}


@app.post("/query")
def handle_user_query(query_request: QueryRequest):
    user_query_text = query_request.user_query_text
    limit = query_request.limit

    try:
        keywords = extract_relevant_keywords(user_query_text)
        search_results = search_by_keywords_or_author(milvus_wrapper, keywords, user_query_text, limit=limit)
        
        if search_results:
            llm = LLMBackendCloud()
            prompt = prepare_prompt(user_query_text, search_results)
            response = llm.generate_response(prompt)

            # Parse the response into the expected format
            formatted_results = [
                {
                    "assetId": result["id"]
                }
                for result in search_results
            ]

            return formatted_results  # This returns the response in the correct JSON format
        else:
            return {
                "query": user_query_text,
                "message": "No relevant assets found for the given query."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        milvus_wrapper.close()

@app.get("/")
async def root():
    return {"message": "API is running"}
