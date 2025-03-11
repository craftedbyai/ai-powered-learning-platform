import streamlit as st
from utils.chat_with_pdf import (
    add_to_pinecone,
    query_pinecone,
    process_document,
    call_llm,
    re_rank_cross_encoders,
    setup_gemini,
    stream_llm_response,
)


def main():
    st.title("Chat with PDF")
    st.write("Upload a PDF and ask questions about its content")

    setup_gemini()

    uploaded_file = st.file_uploader(
        "**📑 Upload PDF files for Q&A**", type=["pdf"], accept_multiple_files=False
    )

    process = st.button(
        "⚡️ Process",
    )

    if uploaded_file and process:
        try:
            with st.spinner("Processing document..."):
                # Display some information about the file for debugging
                st.info(
                    f"File name: {uploaded_file.name}, Size: {uploaded_file.size} bytes"
                )

                # Normalize the filename for Pinecone
                normalize_uploaded_file_name = uploaded_file.name.translate(
                    str.maketrans({"-": "_", ".": "_", " ": "_"})
                )

                # Process the document
                all_splits = process_document(uploaded_file)

                if all_splits and len(all_splits) > 0:
                    st.success(f"Document processed into {len(all_splits)} chunks")
                    add_to_pinecone(all_splits, normalize_uploaded_file_name)
                    st.success(
                        f"Document '{uploaded_file.name}' processed and added to the knowledge base!"
                    )
                else:
                    st.error("No content was extracted from the document")
        except Exception as e:
            st.error(f"Error during document processing: {e}")
            import traceback

            st.error(traceback.format_exc())

    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button(
        "🔥 Ask",
    )

    if ask and prompt:
        if not uploaded_file:
            st.warning("Please upload a document first!")
        else:
            with st.spinner("Generating answer..."):
                try:
                    results = query_pinecone(
                        prompt,
                        uploaded_file.name.translate(
                            str.maketrans({"-": "_", ".": "_", " ": "_"})
                        ),
                    )
                    if not results["documents"] or len(results["documents"][0]) == 0:
                        st.warning(
                            "No relevant information found. Please try a different question or upload a relevant document."
                        )
                    else:

                        relevant_text, relevant_ids = re_rank_cross_encoders(
                            results["documents"][0], prompt
                        )

                        # with st.expander("Sources"):
                        #     for idx, doc_id in enumerate(relevant_ids):
                        #         st.write(
                        #             f"**Source {idx+1}:** Chunk from extracted text with id {doc_id}."
                        #         )

                        st.subheader("Answer:")
                        response_placeholder = st.empty()
                        stream = stream_llm_response(relevant_text, prompt)
                        full_response = ""
                        for chunk in stream:
                            # Since stream_llm_response now yields text strings directly
                            full_response += chunk
                            response_placeholder.markdown(full_response)
                except Exception as e:
                    st.error(f"Error during question answering: {e}")


if __name__ == "__main__":
    main()
