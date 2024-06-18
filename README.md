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

