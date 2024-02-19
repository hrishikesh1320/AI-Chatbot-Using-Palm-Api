import streamlit as st
import google. generativeai as palm 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

palm.configure(api_key="your_api_key")
defaults = {
    'model': 'models/text-bison-001',
    'temperature': 0.7,
    'candidate_count': 1,
    'top_k': 40,
    'top_p': 0.95,
    'max_output_tokens': 1024,
    'stop_sequences': [],
}

# Pre-defined set of example questions
example_questions = [
    "What are the basic elements of a contract?",
    "How does intellectual property law protect inventions?",
    "Can you explain the concept of negligence in tort law?",
    # Add more example questions as needed
]

def generate_response(prompt):
    response = palm.generate_text(**defaults, prompt=prompt)
    return response.result

def recommend_questions(user_input):
    vectorizer = TfidfVectorizer()
    all_questions = example_questions + [user_input]
    tfidf_matrix = vectorizer.fit_transform(all_questions)
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    recommended_indices = similarity_scores.argsort()[0][::-1]  # Sort in descending order

    recommended_questions = [example_questions[idx] for idx in recommended_indices[:3]]
    return recommended_questions

def main():
    st.title("AI-powered Legal Documentation Assistant")
    st.write("Enter your question below:")

    user_input = st.text_input("Question:")
    if st.button("Get Answer"):
        answer = generate_response(user_input)
        st.write("Answer:", answer)

        # Get recommended questions based on user input
        recommended = recommend_questions(user_input)
        st.write("Recommended Questions:")
        for i, question in enumerate(recommended, start=1):
            st.write(f"{i}. {question}")

if __name__ == "__main__":
    main()

