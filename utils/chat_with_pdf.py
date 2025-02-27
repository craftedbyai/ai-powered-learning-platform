import os
import tempfile
import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import google.generativeai as genai
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

# System prompt for the LLM
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.
context will be passed as "Context:"
user question will be passed as "Question:"
To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question. using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context. Do not include any external knowledge or assumptions not present in the given text.
Please be very detailed in your answer. Use at least 5 sentences.
Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. Ensure relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.
Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""


# Configure Gemini API
def setup_gemini():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Missing Gemini API key in the Streamlit secrets. Please add it.")
        st.stop()
    genai.configure(api_key=api_key)
    return True


# Process PDF document using PyMuPDF
def process_document(uploaded_file: UploadedFile) -> list[Document]:
    try:
        with st.spinner("Processing document..."):
            temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
            temp_file.write(uploaded_file.read())
            loader = PyMuPDFLoader(temp_file.name)
            docs = loader.load()
            os.unlink(temp_file.name)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,  # Smaller chunksize helps
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", "?", "!", " ", ""],
            )
            splits = text_splitter.split_documents(docs)
            return splits
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return []


# Set up ChromaDB vector store with sentence transformers
def get_vector_collection(force_reset=False) -> chromadb.Collection:
    try:
        # Ensure directory exists
        os.makedirs("./demo-rag-chroma", exist_ok=True)

        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2"
        )

        # Create persistent client
        chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")

        # Force deletion of collection if requested
        if force_reset:
            try:
                chroma_client.delete_collection(name="rag_app")
                st.info("Vector database has been reset for new document.")
            except Exception as e:
                # Collection might not exist, that's fine
                pass

        # Try to get or create collection
        try:
            collection = chroma_client.get_collection(
                name="rag_app", embedding_function=embedding_function
            )
        except Exception:
            # Create if it doesn't exist
            collection = chroma_client.create_collection(
                name="rag_app",
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"},
            )

        return collection
    except Exception as e:
        st.error(f"Error setting up vector collection: {e}")
        raise


# Add document chunks to vector store
def add_to_vector_collection(all_splits: list[Document], file_name: str):
    try:
        # Get collection with force reset to clear previous data
        collection = get_vector_collection(force_reset=True)

        documents, metadatas, ids = [], [], []
        for idx, split in enumerate(all_splits):
            documents.append(split.page_content)
            metadatas.append(split.metadata)
            ids.append(f"{file_name}_{idx}")

        # Progress indicator
        progress_text = st.empty()
        progress_text.write(f"Processing {len(documents)} document chunks...")

        # Upsert to vector store
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        progress_text.empty()
        st.success(f"Successfully added {len(documents)} chunks to the vector store!")
    except Exception as e:
        st.error(f"Error adding documents to vector store: {e}")


# Query the vector collection
def query_collection(prompt: str, n_results: int = 10):
    try:
        collection = get_vector_collection()
        results = collection.query(query_texts=[prompt], n_results=n_results)
        return results
    except Exception as e:
        st.error(f"Error querying vector store: {e}")
        return {"documents": [], "ids": [], "metadatas": [], "distances": []}


# Re-rank documents using cross-encoder
def re_rank_cross_encoders(documents: list[str], prompt: str) -> tuple[str, list[int]]:
    try:
        relevant_text = ""
        relevant_text_ids = []
        encoder_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )  # Can try to improve cross encoder
        ranks = encoder_model.rank(
            prompt, documents, top_k=min(5, len(documents))
        )  # More
        for rank in ranks:
            relevant_text += documents[rank["corpus_id"]] + "\n\n"
            relevant_text_ids.append(rank["corpus_id"])
        return relevant_text, relevant_text_ids
    except Exception as e:
        st.error(f"Error in re-ranking: {e}")
        return "\n\n".join(documents[: min(5, len(documents))]), list(
            range(min(5, len(documents)))  # More
        )


# Call Gemini model with context and question
def call_llm(context: str, prompt: str):
    try:
        setup_gemini()
        # Create the model
        model = genai.GenerativeModel("gemini-1.5-flash")
        # Generate response
        response = model.generate_content(
            f"System: {system_prompt}\n\nContext: {context}\n\nQuestion: {prompt}"
        )
        return response.text
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return "I encountered an error generating a response. Please try again."


# Stream response (for better user experience)
def stream_llm_response(context: str, prompt: str):
    try:
        setup_gemini()
        model = genai.GenerativeModel("gemini-1.5-flash")
        # Generate streaming response
        response = model.generate_content(
            f"System: {system_prompt}\n\nContext: {context}\n\nQuestion: {prompt}",
            stream=True,
        )
        for chunk in response:
            # Simply yield the text string, not the object
            yield chunk.text if hasattr(chunk, "text") and chunk.text else ""
    except Exception as e:
        st.error(f"Error setting up Gemini API streaming: {e}")
        yield "I encountered an error generating a response. Please try again."


# Main Streamlit app
def main():
    st.title("Document Q&A with Google Gemini")
    st.write("Upload a PDF document and ask questions about its content")

    # Initialize session state for tracking uploaded files
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a PDF document", type="pdf", accept_multiple_files=False
    )

    # Process uploaded file
    if uploaded_file:
        # Reset session state and process new file, regardless of whether
        # we've seen this filename before (could be different content)
        with st.spinner("Processing document..."):
            # Clear uploaded files list to force reset
            st.session_state.uploaded_files = []

            all_splits = process_document(uploaded_file)
            if all_splits:
                add_to_vector_collection(all_splits, uploaded_file.name)
                st.session_state.uploaded_files.append(uploaded_file.name)
                st.success(
                    f"Document '{uploaded_file.name}' processed and added to the knowledge base!"
                )

    # Question input section
    st.subheader("Ask a question")
    query = st.text_input("Enter your question about the document:")
    if st.button("Get Answer") and query:
        if not st.session_state.uploaded_files:
            st.warning("Please upload a document first!")
        else:
            with st.spinner("Searching for relevant information..."):
                # Query vector store
                results = query_collection(query)
                if not results["documents"] or len(results["documents"][0]) == 0:
                    st.warning(
                        "No relevant information found. Please try a different question or upload a relevant document."
                    )
                else:
                    # Re-rank results
                    relevant_text, relevant_ids = re_rank_cross_encoders(
                        results["documents"][0], query
                    )
                    # Display citations
                    with st.expander("Sources"):
                        for idx, doc_id in enumerate(relevant_ids):
                            original_id = results["ids"][0][doc_id]
                            metadata = results["metadatas"][0][doc_id]
                            st.write(f"**Source {idx+1}:** {original_id}")
                            if "page" in metadata:
                                st.write(f"Page: {metadata['page']}")

                    # Get answer from LLM
                    st.subheader("Answer:")
                    # Use streaming for better UX
                    response_placeholder = st.empty()
                    full_response = ""
                    # Stream the response
                    for chunk in stream_llm_response(relevant_text, query):
                        full_response += chunk
                        response_placeholder.markdown(full_response)

                    # If streaming fails, fall back to non-streaming
                    if not full_response:
                        response = call_llm(relevant_text, query)
                        response_placeholder.markdown(response)


if __name__ == "__main__":
    main()
