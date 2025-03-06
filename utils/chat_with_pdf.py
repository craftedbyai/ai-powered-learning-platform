import os
import time
import tempfile
import streamlit as st

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
"""


# Configure Gemini API
def setup_gemini():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Missing Gemini API key in the Streamlit secrets. Please add it.")
        st.stop()
    genai.configure(api_key=api_key)
    return True


# Setup Pinecone client
def setup_pinecone():
    api_key = st.secrets.get("PINECONE_API_KEY")

    if not api_key:
        st.error("Missing Pinecone API key in the Streamlit secrets. Please add it.")
        st.stop()

    try:
        # Initialize Pinecone client with API key (new SDK style)
        pinecone_client = Pinecone(api_key=api_key)
        return pinecone_client
    except Exception as e:
        st.error(f"Error initializing Pinecone: {e}")
        st.stop()
        return None


# Process PDF document using PyMuPDF
def process_document(uploaded_file: UploadedFile) -> list[Document]:
    try:
        with st.spinner("Processing document..."):
            # Reset file pointer to the beginning
            uploaded_file.seek(0)

            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
            temp_file_path = temp_file.name

            # Write content to the temporary file and close it properly
            temp_file.write(uploaded_file.getvalue())
            temp_file.flush()
            temp_file.close()

            # Log for debugging
            file_size = os.path.getsize(temp_file_path)
            st.info(f"Temporary file created: {temp_file_path} ({file_size} bytes)")

            # Load the document
            loader = PyMuPDFLoader(temp_file_path)
            docs = loader.load()

            # Clean up temp file after loading
            os.unlink(temp_file_path)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", "?", "!", " ", ""],
            )
            splits = text_splitter.split_documents(docs)
            return splits
    except Exception as e:
        st.error(f"Error processing document: {e}")
        # For debugging, print the full stack trace
        import traceback

        st.error(traceback.format_exc())
        return []


# Add document chunks to Pinecone
# Get or create Pinecone index
def get_pinecone_index(force_reset=False):
    try:
        pinecone_client = setup_pinecone()
        index_name = "rag-app"
        embedding_dimension = 768  # Dimension for all-mpnet-base-v2 model

        # List all indexes
        indexes = pinecone_client.list_indexes()
        index_names = [index.name for index in indexes]

        # Check if index exists
        index_exists = index_name in index_names

        # Create index if it doesn't exist
        if not index_exists:
            try:
                st.info("Creating new Pinecone index...")
                pinecone_client.create_index(
                    name=index_name,
                    dimension=embedding_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                # Wait for index to be initialized
                ready = False
                attempts = 0
                while not ready and attempts < 10:
                    time.sleep(5)  # Wait 5 seconds between checks
                    try:
                        index_info = pinecone_client.describe_index(index_name)
                        if index_info.status.ready:
                            ready = True
                        else:
                            attempts += 1
                    except:
                        attempts += 1
            except Exception as e:
                st.error(f"Error creating Pinecone index: {e}")

        # Get index
        index = pinecone_client.Index(index_name)

        # If force reset and index exists, delete all vectors instead of deleting the index
        # if force_reset and index_exists:
        #     st.info("Resetting vector database for new document...")
        #     try:
        #         # Delete all vectors in the index by using a delete_all operation
        #         index.delete(delete_all=True)
        #         time.sleep(2)  # Allow time for deletion to complete
        #     except Exception as e:
        #         st.error(f"Error deleting vectors from index: {e}")

        return index

    except Exception as e:
        st.error(f"Error setting up Pinecone index: {e}")
        raise


# Initialize Sentence Transformer model
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-mpnet-base-v2")


# Generate embeddings for text
def generate_embeddings(texts):
    model = get_embedding_model()
    return model.encode(texts).tolist()


# Add document chunks to Pinecone
def add_to_pinecone(all_splits: list[Document], file_name: str):
    try:
        # Get index with force reset to clear previous data
        index = get_pinecone_index(force_reset=True)

        # Prepare data for batch upsert
        batch_size = 100  # Reasonable batch size for Pinecone
        total_chunks = len(all_splits)

        # Progress indicator
        progress_text = st.empty()
        progress_text.write(f"Processing {total_chunks} document chunks...")

        try:
            for i in range(0, total_chunks, batch_size):
                # Get current batch
                batch_splits = all_splits[i : min(i + batch_size, total_chunks)]

                # Extract text for embeddings
                texts = [split.page_content for split in batch_splits]

                # Generate embeddings
                embeddings = generate_embeddings(texts)

                # Prepare vectors for upsert
                vectors = []
                for j, (split, embedding) in enumerate(zip(batch_splits, embeddings)):
                    vector_id = f"{file_name}_{i+j}"
                    # Create upsert record with proper format for Pinecone
                    vectors.append(
                        {
                            "id": vector_id,
                            "values": embedding,
                            "metadata": {
                                "text": split.page_content,
                                "page": split.metadata.get("page", 0),
                                "source": file_name,
                            },
                        }
                    )

                # Upsert to Pinecone - handle potential rate limits with retries
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        index.upsert(vectors=vectors)
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            time.sleep(2**attempt)  # Exponential backoff
                        else:
                            raise e

                progress_text.write(
                    f"Processed {min(i+batch_size, total_chunks)}/{total_chunks} chunks..."
                )

            progress_text.empty()
            st.success(f"Successfully added {total_chunks} chunks to the vector store!")

        except Exception as e:
            st.error(f"Error in Pinecone indexing process: {e}")

    except Exception as e:
        st.error(f"Error setting up Pinecone for document upload: {e}")


# Query the Pinecone index
def query_pinecone(prompt: str, pdf_name: str, n_results: int = 10):
    try:
        index = get_pinecone_index()

        # Generate embedding for the query
        query_embedding = generate_embeddings([prompt])[0]

        # Query Pinecone
        query_results = index.query(
            vector=query_embedding,
            top_k=n_results,
            include_metadata=True,
            filter={"source": {"$eq": pdf_name}},
        )

        # Format results similar to ChromaDB output for compatibility
        documents = [match.metadata.get("text", "") for match in query_results.matches]
        ids = [match.id for match in query_results.matches]
        metadatas = [
            {k: v for k, v in match.metadata.items() if k != "text"}
            for match in query_results.matches
        ]
        distances = [
            1 - match.score for match in query_results.matches
        ]  # Convert cosine similarity to distance

        return {
            "documents": [documents],
            "ids": [ids],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")
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
                add_to_pinecone(all_splits, uploaded_file.name)
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
                results = query_pinecone(query)
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
