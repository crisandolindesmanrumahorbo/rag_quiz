import streamlit as st
import chromadb
import requests
import PyPDF2
import docx
import io
from typing import List, Dict, Any
import time


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def generate(self, model: str, prompt: str, system: str = "") -> str:
        """Generate text using Ollama API"""
        url = f"{self.base_url}/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False
        }

        try:
            response = requests.post(url, json=data, timeout=120)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            st.error(f"Error calling Ollama: {str(e)}")
            return ""

    def embed(self, model: str, text: str) -> List[float]:
        """Get embeddings using Ollama API"""
        url = f"{self.base_url}/api/embeddings"
        data = {
            "model": model,
            "prompt": text
        }

        try:
            response = requests.post(url, json=data, timeout=60)
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            st.error(f"Error getting embeddings: {str(e)}")
            return []


class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return ""

    @staticmethod
    def extract_text_from_docx(file_content: bytes) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(io.BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error processing DOCX: {str(e)}")
            return ""

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks


class RAGQuizGenerator:
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection_name = "book_knowledge"

    def initialize_collection(self):
        """Initialize or get ChromaDB collection"""
        try:
            # Try to get existing collection
            collection = self.chroma_client.get_collection(
                name=self.collection_name)
        except:
            # Create new collection if it doesn't exist
            collection = self.chroma_client.create_collection(
                name=self.collection_name)
        return collection

    def add_documents_to_db(self, documents: List[str], progress_bar=None):
        """Add documents to vector database"""
        collection = self.initialize_collection()

        for i, doc in enumerate(documents):
            if progress_bar:
                progress_bar.progress((i + 1) / len(documents))

            # Get embeddings
            embedding = self.ollama_client.embed("nomic-embed-text:v1.5", doc)
            if embedding:
                collection.add(
                    documents=[doc],
                    embeddings=[embedding],
                    ids=[f"doc_{i}_{int(time.time())}"]
                )

    def search_relevant_content(self, query: str, n_results: int = 3) -> List[str]:
        """Search for relevant content in the database"""
        collection = self.initialize_collection()

        # Get query embedding
        query_embedding = self.ollama_client.embed(
            "nomic-embed-text:v1.5", query)
        if not query_embedding:
            return []

        # Search for similar documents
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        return results["documents"][0] if results["documents"] else []

    def generate_quiz_question(self, topic: str = "", difficulty: str = "medium") -> Dict[str, Any]:
        """Generate a quiz question based on the knowledge base"""

        # Search for relevant content
        search_query = topic if topic else "general knowledge from the book"
        relevant_docs = self.search_relevant_content(search_query, n_results=2)

        if not relevant_docs:
            return {"error": "No relevant content found in the knowledge base"}

        # Combine relevant documents
        context = "\n\n".join(relevant_docs)

        # Create prompt for quiz generation
        system_prompt = f"""You are an expert quiz generator. Create a multiple choice question based on the provided context.

Requirements:
- Generate 1 question with 4 options (A, B, C, D)
- Difficulty level: {difficulty}
- Include the correct answer
- Provide a brief explanation
- Focus on {topic if topic else 'important facts from the content'}

Format your response EXACTLY like this:
QUESTION: [Your question here]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]
ANSWER: [Letter of correct answer]
EXPLANATION: [Brief explanation of why this is correct]"""

        user_prompt = f"""Based on this context from the book, generate a quiz question:

{context}

Create a {difficulty} level multiple choice question."""

        # Generate the quiz question
        response = self.ollama_client.generate(
            model="llama3.1:8b",
            system=system_prompt,
            prompt=user_prompt
        )

        return self.parse_quiz_response(response)

    def parse_quiz_response(self, response: str) -> Dict[str, Any]:
        """Parse the generated quiz response"""
        try:
            lines = response.strip().split('\n')
            quiz_data = {}

            for line in lines:
                line = line.strip()
                if line.startswith('QUESTION:'):
                    quiz_data['question'] = line.replace(
                        'QUESTION:', '').strip()
                elif line.startswith('A.'):
                    quiz_data['option_a'] = line.replace('A.', '').strip()
                elif line.startswith('B.'):
                    quiz_data['option_b'] = line.replace('B.', '').strip()
                elif line.startswith('C.'):
                    quiz_data['option_c'] = line.replace('C.', '').strip()
                elif line.startswith('D.'):
                    quiz_data['option_d'] = line.replace('D.', '').strip()
                elif line.startswith('ANSWER:'):
                    quiz_data['correct_answer'] = line.replace(
                        'ANSWER:', '').strip()
                elif line.startswith('EXPLANATION:'):
                    quiz_data['explanation'] = line.replace(
                        'EXPLANATION:', '').strip()

            return quiz_data
        except Exception as e:
            return {"error": f"Error parsing response: {str(e)}", "raw_response": response}


def main():
    st.set_page_config(page_title="RAG Quiz Generator",
                       page_icon="📚", layout="wide")

    st.title("📚 RAG Quiz Generator")
    st.write("Upload your school book and generate quiz questions!")

    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGQuizGenerator()

    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("📁 Document Upload")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'txt'],
            help="Upload your school book (PDF, DOCX, or TXT)"
        )

        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")

            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    # Extract text based on file type
                    if uploaded_file.type == "application/pdf":
                        text = DocumentProcessor.extract_text_from_pdf(
                            uploaded_file.read())
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        text = DocumentProcessor.extract_text_from_docx(
                            uploaded_file.read())
                    else:  # txt
                        text = str(uploaded_file.read(), "utf-8")

                    if text:
                        # Chunk the text
                        chunks = DocumentProcessor.chunk_text(text)
                        st.write(f"Created {len(chunks)} text chunks")

                        # Add to vector database
                        progress_bar = st.progress(0)
                        st.session_state.rag_system.add_documents_to_db(
                            chunks, progress_bar)

                        st.success(
                            "Document processed and added to knowledge base!")
                        st.session_state.document_processed = True
                    else:
                        st.error("Could not extract text from the document")

        st.header("⚙️ Quiz Settings")
        difficulty = st.selectbox(
            "Difficulty Level", ["easy", "medium", "hard"])
        topic = st.text_input("Specific Topic (optional)",
                              placeholder="e.g., Indonesian History")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("🎯 Generate Quiz")

        if st.button("Generate New Question", type="primary"):
            if not hasattr(st.session_state, 'document_processed'):
                st.warning("Please upload and process a document first!")
            else:
                with st.spinner("Generating quiz question..."):
                    quiz_data = st.session_state.rag_system.generate_quiz_question(
                        topic, difficulty)
                    st.session_state.current_quiz = quiz_data

        # Display generated quiz
        if 'current_quiz' in st.session_state:
            quiz = st.session_state.current_quiz

            if 'error' in quiz:
                st.error(f"Error: {quiz['error']}")
                if 'raw_response' in quiz:
                    st.text("Raw response:")
                    st.text(quiz['raw_response'])
            else:
                st.subheader("📝 Quiz Question")
                st.write(f"**Question:** {quiz.get('question', 'N/A')}")

                # Display options
                st.write("**Options:**")
                for letter in ['a', 'b', 'c', 'd']:
                    option_key = f'option_{letter}'
                    if option_key in quiz:
                        st.write(f"{letter.upper()}. {quiz[option_key]}")

                # User answer selection
                user_answer = st.radio(
                    "Your Answer:",
                    ['A', 'B', 'C', 'D'],
                    key="user_answer"
                )

                if st.button("Submit Answer"):
                    correct = quiz.get('correct_answer', '').upper()
                    if user_answer == correct:
                        st.success("✅ Correct!")
                    else:
                        st.error(f"❌ Wrong! The correct answer is {correct}")

                    st.info(
                        f"**Explanation:** {quiz.get('explanation', 'No explanation available')}")

    with col2:
        st.header("📊 Knowledge Base Status")

        # Check if ChromaDB has content
        try:
            collection = st.session_state.rag_system.initialize_collection()
            count = collection.count()
            st.metric("Documents in Knowledge Base", count)

            if count > 0:
                st.success("Knowledge base is ready!")
            else:
                st.warning("No documents in knowledge base yet.")
        except Exception as e:
            st.error(f"Error checking knowledge base: {str(e)}")

        st.header("💡 Tips")
        st.write("""
        - Upload your school book in PDF, DOCX, or TXT format
        - The system will chunk your document and create embeddings
        - Specify topics for targeted questions
        - Try different difficulty levels
        - The AI uses the book content to generate relevant questions
        """)

        st.header("🔧 System Requirements")
        st.write("""
        **Required Ollama Models:**
        - `llama3.1:8b` (for quiz generation)
        - `nomic-embed-text:v1.5` (for embeddings)
        
        **Make sure both models are pulled:**
        ```bash
        ollama pull llama3.1:8b
        ollama pull nomic-embed-text:v1.5
        ```
        """)


if __name__ == "__main__":
    main()
