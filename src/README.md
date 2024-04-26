You can change and customize the code to test different models and configurations.

### Embeddings models

For this project, I tried 3 different embeddings model to check performance:

#### HuggingFaceHubEmbeddings()

```python
from langchain_community.embeddings import HuggingFaceHubEmbeddings
   
    ...
    vectore_store = Chroma.from_documents(document_chunks, HuggingFaceHubEmbeddings())
```

#### phi3 with ollama
```python
from langchain_community.embeddings import OllamaEmbeddings
   
    ...
    embeddings = OllamaEmbeddings(model='phi3')
    vectore_store = Chroma.from_documents(document_chunks, embeddings)
```

#### nomic-embed-text with ollama

```bash
ollama run nomic-embed-text
```

```python
from langchain_community.embeddings import OllamaEmbeddings
   
    ...
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    vectore_store = Chroma.from_documents(document_chunks, embeddings)
```

### LLm

I used `phi-3` model with ollama.

```bash
ollama run phi3
```
