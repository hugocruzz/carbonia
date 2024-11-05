import os
import pandas as pd
from openai import OpenAI
import re
import unicodedata
from .utils import check_and_normalize_series

def embed(data, embedding_column_name=None, embedding_model="text-embedding-ada-002", api_key=None):
    """Embed a string, list, or dataframe using OpenAI embeddings."""
    
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    client = OpenAI(api_key=api_key)

    def get_embeddings(texts, model="text-embedding-ada-002"):
        texts = [text.replace("\n", " ") for text in texts]
        response = client.embeddings.create(input=texts, model=model)
        return [data.embedding for data in response.data]

    if isinstance(data, str):
        return get_embeddings([data], model=embedding_model)[0]
    
    elif isinstance(data, list):
        return get_embeddings(data, model=embedding_model)
    
    elif isinstance(data, pd.DataFrame):
        def batch_embeddings(texts, batch_size=1500, export_chunk=False):
            total_batches = (len(texts) + batch_size - 1) // batch_size
            embeddings = []

            for i in range(0, len(texts), batch_size):
                current_batch = (i // batch_size) + 1
                print(f"Processing batch {current_batch} of {total_batches}...")
                batch = texts[i:i+batch_size]
                batch_embeddings = get_embeddings(batch, model=embedding_model)
                if export_chunk:
                    pd.DataFrame(batch_embeddings).to_csv(f"data/results/embeddings_{current_batch}.csv")

                embeddings.extend(batch_embeddings)
            
            print("All batches processed.")
            return embeddings
        if embedding_column_name is None:
            embedding_column_name = data.columns
        
        combined_column_name = "combined"
        output_embedding_name = "embedding"
        
        data[embedding_column_name] = data[embedding_column_name].fillna("")
        if isinstance(embedding_column_name, list):
            data[combined_column_name] = data[embedding_column_name].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        else:
            data[combined_column_name] = data[embedding_column_name].astype(str) 
        df_unique = data[[combined_column_name]].drop_duplicates()
        normalized_unique_df = check_and_normalize_series(df_unique[combined_column_name])
        df_unique[output_embedding_name] = batch_embeddings(normalized_unique_df, batch_size=1500, export_chunk=False)
        df_final = data.merge(df_unique, left_on=combined_column_name, right_on=combined_column_name, how="left")
        return df_final
    
    else:
        raise ValueError("Unsupported data type. Please provide a string, list, or pandas DataFrame.")
    
