# llm-rag


We will use pipenv for dependency management. Let's install it: 

```bash
pip install pipenv
```

Install the packages

```bash
pipenv install tqdm notebook==7.1.2 openai elasticsearch
```



If you use OpenAI, let's put the key to an env variable:


Getting the key

* Sign up at https://platform.openai.com/ if you don't have an account
* Go to https://platform.openai.com/api-keys
* Create a new key, copy it 


To manage keys, we can use direnv:

```bash
sudo apt update
sudo apt install direnv 
direnv hook bash >> ~/.bashrc
```
After direnv hook -> When you open new terminal it'll be
```bash
direnv: loading /workspaces/llm-rag/.envrc
direnv: export +OPENAI_API_KEY
```
Create / edit `.envrc` in your project directory:

```bash
export OPENAI_API_KEY='sk-proj-key'
```


Initialize Jupyter Notebook in new terminal

```bash
pipenv run jupyter notebook
```

In another terminal, run elasticsearch with docker:

```bash
docker run -it \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```

Verify that ES is running

```bash
curl http://localhost:9200
```

You should get something like this:

```json
{
  "name" : "084d28d96f18",
  "cluster_name" : "docker-cluster",
  "cluster_uuid" : "_lS2dyasTQqmavMDdf131A",
  "version" : {
    "number" : "8.4.3",
    "build_flavor" : "default",
    "build_type" : "docker",
    "build_hash" : "42f05b9372a9a4a470db3b52817899b99a76ee73",
    "build_date" : "2022-10-04T07:17:24.662462378Z",
    "build_snapshot" : false,
    "lucene_version" : "9.3.0",
    "minimum_wire_compatibility_version" : "7.17.0",
    "minimum_index_compatibility_version" : "7.0.0"
  },
  "tagline" : "You Know, for Search"
}
```

# Retrieval

RAG consists of multiple components, and the first is R - "retrieval". For retrieval, we need a search system. In our example, we will use elasticsearch for searching. 

## Searching in the documents

Create a nootebook "elastic-rag" or something like that. We will use it for our experiments

First, we need to download the docs:

```bash
wget https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json
```

Let's load the documents

```python
import json

with open('./documents.json', 'rt') as f_in:
    documents_file = json.load(f_in)

documents = []

for course in documents_file:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)
```




Now we'll index these documents with elastic search

First initiate the connection and check that it's working:

```python
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")
es.info()
```

You should see the same response as earlier with `curl`.

Before we can index the documents, we need to create an index (an index in elasticsearch is like a table in a "usual" databases):


index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"} 
        }
    }
}

index_name = "course-questions"
response = es.indices.create(index=index_name, body=index_settings)

response


Now we're ready to index all the documents:

```python
from tqdm.auto import tqdm

for doc in tqdm(documents):
    es.index(index=index_name, document=doc)
```


## Retrieving the docs

```python
user_question = "How do I join the course after it has started?"

search_query = {
    "size": 5,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": user_question,
                    "fields": ["question^3", "text", "section"],
                    "type": "best_fields"
                }
            },
            "filter": {
                "term": {
                    "course": "data-engineering-zoomcamp"
                }
            }
        }
    }
}
```

This query:

* Retrieves top 5 matching documents.
* Searches in the "question", "text", "section" fields, prioritizing "question".
* Matches user query "How do I join the course after it has started?".
* Shows results only for the "data-engineering-zoomcamp" course.

Let's see the output:

```python
response = es.search(index=index_name, body=search_query)

for hit in response['hits']['hits']:
    doc = hit['_source']
    print(f"Section: {doc['section']}\nQuestion: {doc['question']}\nAnswer: {doc['text']}\n\n")
```

## Cleaning it

We can make it cleaner by putting it into a function:

```python
def retrieve_documents(query, index_name="course-questions", max_results=5):
    es = Elasticsearch("http://localhost:9200")
    
    search_query = {
        "size": max_results,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        }
    }
    
    response = es.search(index=index_name, body=search_query)
    documents = [hit['_source'] for hit in response['hits']['hits']]
    return documents
```

And print the answers:

```python
user_question = "How do I join the course after it has started?"

response = retrieve_documents(user_question)

for doc in response:
    print(f"Section: {doc['section']}\nQuestion: {doc['question']}\nAnswer: {doc['text']}\n\n")
```

# Generation - Answering questions

Now let's do the "G" part - generation based on the "R" output

## OpenAI

Today we will use OpenAI (it's the easiest to get started with). In the course, we will learn how to use open-source models 

Make sure we have the SDK installed and the key is set.

This is how we communicate with ChatGPT3.5:

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "What's the formula for Energy?"}]
)
print(response.choices[0].message.content)
```

## Building a Prompt

Now let's build a prompt. First, we put all the 
documents together in one string:


```python
context_docs = retrieve_documents(user_question)

context = ""

for doc in context_docs:
    doc_str = f"Section: {doc['section']}\nQuestion: {doc['question']}\nAnswer: {doc['text']}\n\n"
    context += doc_str

context = context.strip()
print(context)
```

Now build the actual prompt:

```python
prompt = f"""
You're a course teaching assistant. Answer the user QUESTION based on CONTEXT - the documents retrieved from our FAQ database. 
Only use the facts from the CONTEXT. If the CONTEXT doesn't contan the answer, return "NONE"

QUESTION: {user_question}

CONTEXT:

{context}
""".strip()
```

Now we can put it to OpenAI API:

```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}]
)
answer = response.choices[0].message.content
answer
```

Note: there are system and user prompts, we can also experiment with them to make the design of the prompt cleaner.

## Cleaning

Now let's put everything together in one function:


```python
def build_context(documents):
    context = ""

    for doc in documents:
        doc_str = f"Section: {doc['section']}\nQuestion: {doc['question']}\nAnswer: {doc['text']}\n\n"
        context += doc_str
    
    context = context.strip()
    return context


def build_prompt(user_question, documents):
    context = build_context(documents)
    return f"""
You're a course teaching assistant.
Answer the user QUESTION based on CONTEXT - the documents retrieved from our FAQ database.
Don't use other information outside of the provided CONTEXT.  

QUESTION: {user_question}

CONTEXT:

{context}
""".strip()

def ask_openai(prompt, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    return answer

def qa_bot(user_question):
    context_docs = retrieve_documents(user_question)
    prompt = build_prompt(user_question, context_docs)
    answer = ask_openai(prompt)
    return answer
```

Now we can ask it:

```python
qa_bot("I'm getting invalid reference format: repository name must be lowercase")

qa_bot("I can't connect to postgres port 5432, my password doesn't work")

qa_bot("how can I run kafka?")
```


# What's next

* Use Open-Souce
* Build an interface, e.g. streamlit
* Deploy it



Refer to https://github.com/alexeygrigorev/llm-rag-workshop/tree/main for full readme for the entire project