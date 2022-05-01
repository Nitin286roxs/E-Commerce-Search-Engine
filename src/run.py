'''
This file loads e-commerce data and prepare text data numerical (text-embdings)
Author: nitinashu1995@gmail.com
'''
import json
import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

INDEX_NAME = "ecom"
INDEX_FILE = "data/ecom/index.json"
DATA_FILE = "data/ecom/Zevi_search_dataset.json"
BATCH_SIZE = 1000
SEARCH_SIZE = 5
GPU_LIMIT = 0.5

##### INDEXING #####
class TextEmbedd:
    '''
    In this class we will generate text embedding data
    '''
    def __init__(self):
        '''
        Intailzation TextEmbedd class
        '''
        self.client = None
        self.embeddings = None
        self.text_ph = None
        self.session = None
        self.product_names = None
        self.docs = None
    def create_session(self):
        '''
        Creating Session
        '''
        print("Creating tensorflow session...")
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = GPU_LIMIT
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.tables_initializer())
        print("Done.")

    def load_pretrained_model(self):
        '''
        In this method, we are loading tensorflow universal-sentence-encoder
        Use tensorflow 1 behavior to match the Universal Sentence Encoder
        examples (https://tfhub.dev/google/universal-sentence-encoder/2).
        '''
        print("Downloading pre-trained embeddings from tensorflow hub...")
        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        self.text_ph = tf.placeholder(tf.string)
        self.embeddings = embed(self.text_ph)
        print("Done.")

    def index_data(self):
        '''
        Data indexing
        '''
        print("Creating the 'ecom' index.")
        self.client = Elasticsearch()
        self.client.indices.delete(index=INDEX_NAME, ignore=[404])

        with open(INDEX_FILE) as index_file:
            source = index_file.read().strip()
            self.client.indices.create(index=INDEX_NAME, body=source)

        count = 0
        self.docs = []
        with open(DATA_FILE,  encoding="utf-8") as data_file:
            data = json.load(data_file)
            print(type(data))
            for line in data:
                #if count == 0:
                #    print(f"line: {line}")
                self.docs.append(line)
                count += 1

                if count % BATCH_SIZE == 0:
                    self.index_batch()
                    self.docs = []
                    print(f"Indexed {count} documents.")

            if self.docs:
                self.index_batch()
                print(f"Indexed {count} documents.")

        self.client.indices.refresh(index=INDEX_NAME)
        print("Done indexing.")

    def index_batch(self):
        '''
        Batching docs
        '''
        self.product_names = [doc["name"] for doc in self.docs]
        title_vectors = self.embed_text(self.product_names)
        requests = []
        for i, doc in enumerate(self.docs):
            request = doc
            request["_op_type"] = "index"
            request["_index"] = INDEX_NAME
            request["name_vector"] = title_vectors[i]
            requests.append(request)
        bulk(self.client, requests)

    def embed_text(self, data):
        '''
        Genearing embedding-text
        '''
        vectors = self.session.run(self.embeddings, feed_dict={self.text_ph: data })
        return [vector.tolist() for vector in vectors]

    def run_query_loop(self):
        '''
        Online Query
        '''
        while True:
            try:
                self.handle_query()
            except (KeyboardInterrupt, UnicodeEncodeError) :
                pass

    def handle_query(self):
        '''
        Seraching query in indexed data
        '''
        query = input("Enter query: ")
        embedding_start = time.time()
        query_vector = self.embed_text([query])[0]
        embedding_time = time.time() - embedding_start
        script_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, doc['name_vector']) + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
        search_start = time.time()
        response = self.client.search(
            index=INDEX_NAME,
            body={
                "size": SEARCH_SIZE,
                "query": script_query,
                "_source": {"includes": ["name",\
                                         "price",\
                                         "rating",\
                                         "popularity",\
                                         "description",\
                                         "url"]}
            }
        )
        search_time = time.time() - search_start
        print()
        print(f'{response["hits"]["total"]["value"]} total hits.')
        print(f'embedding time: {embedding_time * 1000:.2f} ms')
        print(f'search time: {search_time * 1000:.2f} ms')
        for hit in response["hits"]["hits"]:
            print(f'id: {hit["_id"]}, score: {hit["_score"]}')
            print(hit["_source"])
            print()


if __name__ == "__main__":
    # Creating instance of TextEmbedd class
    text_embedd_obj = TextEmbedd()
    # Loading pretrained  model
    text_embedd_obj.load_pretrained_model()
    # Creating tensorflow session
    text_embedd_obj.create_session()
    # Indexing Data
    text_embedd_obj.index_data()
    # Running query loop
    text_embedd_obj.run_query_loop()

    print("Closing tensorflow session...")
    text_embedd_obj.session.close()
    print("Done.")
