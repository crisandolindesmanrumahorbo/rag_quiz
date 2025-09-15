import streamlit as st
import chromadb
import requests
import PyPDF2
import docx
import io
import re
from typing import List, Dict, Any
import time

# TODO
# 1. change pdf lib to unstructured


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

    def generate_quiz_questions(self, num_questions: int = 1, topic: str = "", difficulty: str = "medium") -> List[Dict[str, Any]]:
        """Generate multiple quiz questions based on the knowledge base in one LLM call"""

        # Search for relevant content
        search_query = topic if topic else "general knowledge from the book"
        relevant_docs = self.search_relevant_content(
            search_query, n_results=min(5, num_questions + 2))
        st.warning(f"Topic: {topic} relevant info got {
            relevant_docs}.")

        if not relevant_docs:
            return [{"error": "No relevant content found in the knowledge base"}]

        # Combine relevant documents
        context = "\n\n".join(relevant_docs)

        # Create prompt for multiple quiz generation
        system_prompt = f"""You are an expert quiz generator. Create {num_questions} multiple choice questions based on the provided context.

Requirements:
- Generate exactly {num_questions} questions, each with 4 options (A, B, C, D)
- Difficulty level: {difficulty}
- Include the correct answer for each question
- Provide a brief explanation for each answer
- Focus on {topic if topic else 'important facts from the content'}
- Make sure questions cover different aspects of the content
- Avoid duplicate or very similar questions

Format your response EXACTLY like this for each question:
QUESTION 1: [Your first question here]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]
ANSWER: [Letter of correct answer]
EXPLANATION: [Brief explanation of why this is correct]

QUESTION 2: [Your second question here]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]
ANSWER: [Letter of correct answer]
EXPLANATION: [Brief explanation of why this is correct]

Continue this pattern for all {num_questions} questions."""

        user_prompt = f"""Based on this context from the book, generate {num_questions} different quiz questions:

{context}

Create {num_questions} {difficulty} level multiple choice questions covering different aspects of the content."""

        # Generate the quiz questions
        response = self.ollama_client.generate(
            model="llama3.1:8b",
            system=system_prompt,
            prompt=user_prompt
        )

        return self.parse_multiple_quiz_response(response, num_questions)

    def parse_multiple_quiz_response(self, response: str, expected_questions: int) -> List[Dict[str, Any]]:
        """Parse the generated multiple quiz response"""
        try:
            questions = []
            lines = response.strip().split('\n')
            current_question = {}

            for line in lines:
                line = line.strip()

                # Check for question start (QUESTION 1:, QUESTION 2:, etc.)
                if re.match(r'^QUESTION \d+:', line):
                    # Save previous question if exists
                    if current_question and 'question' in current_question:
                        questions.append(current_question)
                    # Start new question
                    current_question = {}
                    current_question['question'] = re.sub(
                        r'^QUESTION \d+:', '', line).strip()
                elif line.startswith('A.'):
                    current_question['option_a'] = line.replace(
                        'A.', '').strip()
                elif line.startswith('B.'):
                    current_question['option_b'] = line.replace(
                        'B.', '').strip()
                elif line.startswith('C.'):
                    current_question['option_c'] = line.replace(
                        'C.', '').strip()
                elif line.startswith('D.'):
                    current_question['option_d'] = line.replace(
                        'D.', '').strip()
                elif line.startswith('ANSWER:'):
                    current_question['correct_answer'] = line.replace(
                        'ANSWER:', '').strip()
                elif line.startswith('EXPLANATION:'):
                    current_question['explanation'] = line.replace(
                        'EXPLANATION:', '').strip()

            # Add the last question
            if current_question and 'question' in current_question:
                questions.append(current_question)

            # Validate we got the expected number of questions
            if len(questions) != expected_questions:
                st.warning(f"Expected {expected_questions} questions but got {
                           len(questions)}. This might be due to LLM response formatting.")

            return questions if questions else [{"error": "Could not parse any questions from response", "raw_response": response}]

        except Exception as e:
            return [{"error": f"Error parsing response: {str(e)}", "raw_response": response}]


def main():
    st.set_page_config(page_title="RAG Quiz Generator",
                       page_icon="📚", layout="wide")

    st.title("📚 RAG Quiz Generator")
    st.write("Upload your school book and generate quiz questions!")

    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGQuizGenerator()

    # Quiz settings - defined at top level so accessible everywhere
    st.sidebar.header("⚙️ Quiz Settings")
    difficulty = st.sidebar.selectbox(
        "Difficulty Level", ["easy", "medium", "hard"])
    topic = st.sidebar.text_input(
        "Specific Topic (optional)", placeholder="e.g., Indonesian History")
    num_questions = st.sidebar.number_input(
        "Number of Questions", min_value=1, max_value=10, value=1)

    # Sidebar for file upload
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
                        st.write(f"Created {len(chunks)} text {chunks}")

                        # Add to vector database
                        progress_bar = st.progress(0)
                        st.session_state.rag_system.add_documents_to_db(
                            chunks, progress_bar)

                        st.success(
                            "Document processed and added to knowledge base!")
                        st.session_state.document_processed = True
                    else:
                        st.error("Could not extract text from the document")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("🎯 Generate Quiz")

        if st.button("Generate Quiz Questions", type="primary"):
            if not hasattr(st.session_state, 'document_processed') and st.session_state.rag_system.initialize_collection().count() == 0:
                st.warning("Please upload and process a document first!")
            else:
                with st.spinner(f"Generating {num_questions} quiz question(s) in one request..."):
                    # Single LLM call for multiple questions
                    quiz_questions = st.session_state.rag_system.generate_quiz_questions(
                        num_questions, topic, difficulty)

                    st.session_state.quiz_questions = quiz_questions
                    st.session_state.user_answers = [
                        None] * len(quiz_questions)
                    st.session_state.quiz_submitted = [
                        False] * len(quiz_questions)

        # Display generated quiz questions
        if 'quiz_questions' in st.session_state:
            quiz_questions = st.session_state.quiz_questions

            for idx, quiz in enumerate(quiz_questions):
                st.markdown("---")
                st.subheader(f"📝 Question {idx + 1}")

                if 'error' in quiz:
                    st.error(f"Error in question {idx + 1}: {quiz['error']}")
                    if 'raw_response' in quiz:
                        with st.expander("Show raw response"):
                            st.text(quiz['raw_response'])
                else:
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
                        key=f"user_answer_{idx}"
                    )

                    if st.button(f"Submit Answer {idx + 1}", key=f"submit_{idx}"):
                        st.session_state.user_answers[idx] = user_answer
                        st.session_state.quiz_submitted[idx] = True

                    # Show result if submitted
                    if st.session_state.quiz_submitted[idx]:
                        correct = quiz.get('correct_answer', '').upper()
                        user_ans = st.session_state.user_answers[idx]

                        if user_ans == correct:
                            st.success("✅ Correct!")
                        else:
                            st.error(
                                f"❌ Wrong! The correct answer is {correct}")

                        st.info(
                            f"**Explanation:** {quiz.get('explanation', 'No explanation available')}")

            # Show overall score
            if all(st.session_state.quiz_submitted):
                st.markdown("---")
                st.header("🎯 Quiz Results")

                correct_count = 0
                total_questions = len(quiz_questions)

                for i, quiz in enumerate(quiz_questions):
                    if 'error' not in quiz:
                        correct_answer = quiz.get('correct_answer', '').upper()
                        user_answer = st.session_state.user_answers[i]
                        if user_answer == correct_answer:
                            correct_count += 1

                score_percentage = (correct_count / total_questions) * 100
                st.metric("Final Score", f"{
                          correct_count}/{total_questions} ({score_percentage:.1f}%)")

                if score_percentage >= 80:
                    st.balloons()
                    st.success("🎉 Excellent work!")
                elif score_percentage >= 60:
                    st.success("👍 Good job!")
                else:
                    st.warning("📚 Keep studying!")

                if st.button("Generate New Quiz"):
                    # Clear previous quiz data
                    if 'quiz_questions' in st.session_state:
                        del st.session_state.quiz_questions
                    if 'user_answers' in st.session_state:
                        del st.session_state.user_answers
                    if 'quiz_submitted' in st.session_state:
                        del st.session_state.quiz_submitted
                    st.rerun()

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
        - Select 1-10 questions to generate at once
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
