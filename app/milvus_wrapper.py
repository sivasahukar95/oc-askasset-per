from pymilvus import MilvusClient
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import re
import json

class MilvusWrapper:
    def __init__(self, db_path, collection_name):
        """
        Initialize the MilvusWrapper with a database path and collection name.
        """
        self.client = MilvusClient(db_path)  # Initialize the Milvus client
        self.collection_name = collection_name  # Set the collection name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Load a pre-trained model for embeddings
        self.dimension = 384  # Dimension of the embeddings
        self.df = None  # DataFrame placeholder
        self.id_mapping = {}  # To store the mapping between hashed ID and original ID

    def create_collection(self):
        """Create a new collection in Milvus with the specified name and dimension."""
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension
        )

    def delete_all_data(self):
        """Delete all data in the collection by dropping it and creating a new one."""
        try:
            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)
                print(f"Collection '{self.collection_name}' dropped successfully.")
            self.create_collection()
            print(f"Collection '{self.collection_name}' created successfully.")
        except Exception as e:
            print(f"Failed to delete or recreate collection '{self.collection_name}':", e)

    def load_data(self, csv_file_path):
        """Load data from a CSV file and prepare it for insertion into Milvus."""
        self.df = pd.read_csv(csv_file_path)
        # Store the original IDs and map them to their hashed versions
        self.df['hashed_id'] = self.df['id'].apply(lambda x: hash(x) % (10 ** 8))
        self.id_mapping = dict(zip(self.df['hashed_id'], self.df['id']))  # Create the mapping from hashed ID to original ID

        # Combine 'author', 'description' into a 'description' column for embedding generation
        self.df['description'] = self.df.apply(lambda row: f"{self.preprocess_text(row['author'])} {self.preprocess_text(row['description'])}", axis=1)

        # Combine 'author', 'title', and 'description' into a new column for embedding generation
        self.df['author_title_combined'] = self.df.apply(lambda row: f"{self.preprocess_text(row['author'])} {self.preprocess_text(row['title'])} {self.preprocess_text(row['description'])}", axis=1)

        # Generate embeddings for 'description' and 'author_title_combined'
        self.df['description_vector'] = self.df['description'].apply(
            lambda x: self.model.encode(x) if pd.notnull(x) else np.zeros(self.dimension)
        )
        self.df['author_title_vector'] = self.df['author_title_combined'].apply(
            lambda x: self.model.encode(x) if pd.notnull(x) else np.zeros(self.dimension)
        )

        data = [
            {
                "id": row['hashed_id'],  # Store hashed ID for Milvus insertion
                "vector": row['description_vector'].tolist(),
                "author_title_vector": row['author_title_vector'].tolist(),
                "title": row['title'],
                "author": row['author'],
                "description": row['description'],
                "type": row['type']
            }
            for index, row in self.df.iterrows()
        ]
        return data

    def insert_data(self, data):
        """Insert data into the Milvus collection."""
        res = self.client.insert(collection_name=self.collection_name, data=data)
        print("Data Insertion Completed")

    def preprocess_text(self, text):
        """Preprocess text by lowercasing, removing punctuation, and keeping spaces."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        return text.strip()

    def search_by_author(self, author_name, limit=10):
        """Search for assets by a specific author."""
        processed_author = self.preprocess_text(author_name)

        if self.df is None:
            raise ValueError("Data is not loaded. Please load the data first using load_data()")

        filtered_df = self.df[self.df['author'].apply(lambda x: processed_author in self.preprocess_text(x))]
        if len(filtered_df) < limit:
            limit = len(filtered_df)

        results = []
        for _, row in filtered_df.head(limit).iterrows():
            result = {
                "id": row['hashed_id'],  # Return the hashed ID
                "distance": None,
                "title": row['title'],
                "author": row['author'],
                "description": row['description'],
                "type": row['type']
            }
            results.append(result)

        return json.dumps(results, indent=4)

    def search_by_keywords(self, keywords, limit=10):
        """Search by specific technology keywords in description and title."""
        if self.df is None:
            raise ValueError("Data is not loaded. Please load the data first using load_data()")

        filtered_df = self.df[
            (self.df['title'].str.contains('|'.join(keywords), case=False, na=False)) |
            (self.df['description'].str.contains('|'.join(keywords), case=False, na=False))
        ]
        
        if len(filtered_df) < limit:
            limit = len(filtered_df)

        results = []
        for _, row in filtered_df.head(limit).iterrows():
            result = {
                "id": row['hashed_id'],  # Return the hashed ID
                "distance": None,  # Filtering by keywords does not use distance
                "title": row['title'],
                "author": row['author'],
                "description": row['description'],
                "type": row['type']
            }
            results.append(result)

        return json.dumps(results, indent=4)

    def search(self, query_text, limit=3):
        """Perform a vector similarity search across all fields in the Milvus collection."""
        query_vector = self.model.encode(query_text).tolist()

        try:
            res = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                vector_field="vector",
                limit=limit,
                output_fields=["title", "author", "description", "type"]
            )
            return res
        except Exception as e:
            print(f"Failed to perform search:", e)

    def clean_result(self, result):
        """Clean and format search results for readability, converting hashed IDs back to original."""
        cleaned_results = []

        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    entity = item.get('entity', {})
                    original_id = self.id_mapping.get(item.get('id'), "Unknown")  # Retrieve original ID from mapping
                    cleaned_entry = {
                        'id': original_id,  # Use the original ID in the cleaned results
                        'distance': item.get('distance'),
                        'title': entity.get('title'),
                        'author': entity.get('author'),
                        'description': entity.get('description'),
                        'type': entity.get('type')
                    }
                    cleaned_results.append(cleaned_entry)
        return cleaned_results

    def close(self):
        """Close the connection to the Milvus client."""
        self.client.close()
        print("Milvus connection closed.")
