# Quiz Generator Prototype

## How To Run

1. Run the llm which with ollama client with `ollama serve`
2. Run the rag project with `streamlit run rag_quiz_generator_unstructured.py`

## Documentation

[![Watch the video](https://i.sstatic.net/Vp2cE.png)](https://youtu.be/soRS_2VWVAg?si=Ylt1QzAPweM7TB2k)

## How it works

![alt text](<doc/RAG Quiz Generator.jpg>)

## Tools

- Unstructured to extract pdf content
- nomic-embed-text:v1.5
- llama3.2:3b as llm
- chromadb as local vector database

## Why use all of these tools?

Since its only prototype, the goal just to make things work with minimum error
