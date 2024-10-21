# main.py
from milvus_wrapper import MilvusWrapper
from tech_keywords import tech_keywords  # Import tech keywords from the new file
import json
import os
from dotenv import load_dotenv
from ibm_watson_machine_learning.foundation_models import Model
import re

DEFAULT_LLM_PARAMS = {
    'decoding_method': "greedy",
    'min_new_tokens': 1,
    'max_new_tokens': 400,
    'repetition_penalty': 1,
    'random_seed': 42,
}

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
    """Extract the author's name from a user query, while avoiding known tech entities or keywords."""
    # Restrict to specific patterns like "by", "authored by", "associated with", etc. for author extraction.
    match = re.search(r'\b(?:by|of|for|authored by|associated with|related to|belonging to|from|about|connected to|involved in|developed by) ([A-Za-z ]+)', query, re.IGNORECASE)
    if match:
        author_candidate = match.group(1).strip().lower()
        # Ensure the candidate is not in the list of tech keywords
        if author_candidate not in [keyword.lower() for keyword in tech_keywords]:
            return match.group(1).strip()
    
    # Handle possessive form like "Priyanka Mohekar's assets"
    possessive_match = re.search(r'([A-Za-z ]+)(?:\'s) (?:assets|work)', query, re.IGNORECASE)
    if possessive_match:
        author_candidate = possessive_match.group(1).strip().lower()
        if author_candidate not in [keyword.lower() for keyword in tech_keywords]:
            return possessive_match.group(1).strip()
    
    return None


def extract_entities_from_query(query):
    """Extract important keywords or entities like technologies, tools, or tasks."""
    entities = []

    query = re.sub(r'\blist\b|\ball\b|\bfor\b|\bassets\b', '', query, flags=re.IGNORECASE).strip()

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

    return {
        'author': author,
        'entities': entities
    }

# def prepare_prompt(user_query_text, search_results):
#     """Prepare the final prompt to send to the LLM."""
#     return DEFAULT_PROMPT.format(user_query_text=user_query_text, search_results=search_results)

# def prepare_prompt(user_query_text, search_results):
#     """Prepare the final prompt to send to the LLM, including actual results."""
#     results_str = "\n".join([f"ID: {result['id']}, Title: {result['title']}, Author: {result['author']}, Description: {result['description']}" for result in search_results])
#     return f"""
#     You are a helpful, respectful, and honest assistant. Here are the search results:

#     {results_str}

#     Provide a summary of the most relevant result.
#     """

# def prepare_prompt(user_query_text, search_results):
#     """Prepare the final prompt to send to the LLM, including actual results."""
#     results_str = "\n".join([f"ID: {result['id']}, Title: {result['title']}, Author: {result['author']}, Description: {result['description']}" for result in search_results])
#     return f"""
#     You are a helpful, respectful, and honest assistant. Here are the search results:

#     {results_str}

#     Based on the user query: "{user_query_text}", provide a summary of **all** relevant results, including the IDs, titles, and descriptions. Ensure that you include all the results in your response.
#     """

def prepare_prompt(user_query_text, search_results):
    """
    Prepare the final prompt to send to the LLM, ensuring the LLM outputs text
    with ID, title, author, and description included in the final response.
    """
    results_str = "\n\n".join([
        f"{idx + 1}. ID {result['id']}, titled \"{result['title']}\" and created by {result['author']}. "
        f"Description: {result['description']}"
        for idx, result in enumerate(search_results)
    ])

    # Prepare the LLM prompt with clear instructions to return a text summary with all details, including IDs
    return f"""
    You are a helpful, respectful, and honest assistant. The user has requested information based on the following query: "{user_query_text}".

    Below are the search results:

    {results_str}

    Based on the search results, provide a clear and concise text-based response that summarizes the relevant assets. 
    Ensure that the ID, title, author, and description for each result are included in the response. 
    Do not include any JSON, dictionaries, or lists in your response. Only provide a natural language summary.
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
    """
    Searches based on extracted keywords (technology/product) and author names.
    Falls back to vector search if no specific keywords are found.
    """
    author = keywords.get('author')
    entities = keywords.get('entities')

    if author:
        # First, search by author if mentioned
        author_search_result = milvus_wrapper.search_by_author(author, limit=limit)
        cleaned_results = json.loads(author_search_result)
    elif entities:
        # Search by technology keywords if mentioned or general search terms like 'entity extraction'
        keyword_search_result = milvus_wrapper.search_by_keywords(entities, limit=limit)
        cleaned_results = json.loads(keyword_search_result)
    else:
        # Fallback to vector search across all fields
        vector_search_result = milvus_wrapper.search(user_query_text, limit=limit)
        cleaned_results = milvus_wrapper.clean_result(vector_search_result)  # Convert hashed IDs to original

    # Map hashed IDs to original before returning results
    cleaned_results = map_hashed_ids_to_original(milvus_wrapper, cleaned_results)

    return cleaned_results

def main():
    db_path = "./milvus_database.db"
    collection_name = "asset_collection"
    csv_file_path = "./assetdata_jul-22-2024.csv"
    
    # Test user query example
    #user_query_text = "AWS related assets"
    #user_query_text = "S3 related assets"
    #user_query_text = "show me the assets where COS is involved"
    #user_query_text = "show me the results of knowledge graph"
    #user_query_text = "list all assets for entity extraction"
    #user_query_text = "give all assets with entity extraction"
    #user_query_text = "fetch all assets related to entity extraction"
    #user_query_text = "list all assets by Priyanka Mohekar"
    #user_query_text = "list all assets related to Priyanka Mohekar"
    #user_query_text = "retrieve all assets about Priyanka Mohekar"
    #user_query_text = "show all assets associated with Priyanka Mohekar"
    #user_query_text = "list assets by Aditya Mahakali"
    #user_query_text = "list assets for Aditya Mahakali"
    #user_query_text = "give all assets connected to Aditya Mahakali"
    #user_query_text = "give all assets connected to Kanishk Saxena"
    #user_query_text = "list assets by Pankaj Balchandani" 
    #user_query_text = "list assets of Pankaj Balchandani" 
    #user_query_text = "list all assets for Interactive Chatbot"
    #user_query_text = "list all assets by Rishabh Raj, to access an S3 compliant bucket to retrieve its contents."
    #user_query_text = "please showcase the orchestrate related assets"
    #user_query_text = "please showcase the wxo related assets"
    # user_query_text = "please showcase the CP4D related assets"
    user_query_text = "assets related to trulens"

    limit = 3

    # Initialize MilvusWrapper
    milvus_wrapper = MilvusWrapper(db_path, collection_name)
    milvus_wrapper.delete_all_data()

    # Load and insert data
    data = milvus_wrapper.load_data(csv_file_path)
    milvus_wrapper.insert_data(data)

    # Extract relevant keywords (author and tech keywords) from the query
    keywords = extract_relevant_keywords(user_query_text)
    print('keywords')
    print(keywords)

    # Perform the search by author, tech_keywords, or description fallback
    search_results = search_by_keywords_or_author(milvus_wrapper, keywords, user_query_text, limit=limit)
    # print('search_results')
    # print(search_results)

    if search_results:
        # Initialize the LLM to generate a response
        llm = LLMBackendCloud()
        #search_results_json = json.dumps(search_results, indent=4)
        prompt = prepare_prompt(user_query_text, search_results)
        response = llm.generate_response(prompt)

        # Print the search results in the desired format
        for result in search_results:
            print(f"ID: {result['id']}, Title: {result['title']}, Author: {result['author']}, Description: {result['description']}")
        
        print("\nLLM Response:\n", response)
    else:
        print("No relevant assets found for the given query.")

    # Close the Milvus connection
    milvus_wrapper.close()

if __name__ == "__main__":
    main()
