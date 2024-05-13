import pandas as pd
import spacy
import nltk
nltk.download('punkt')
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from spacy.tokens import Doc

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv('cleaned_data.csv', encoding="latin1")

def resolve_coreferences(text):
    doc = nlp(text)
    if doc._.has_coref:
        return doc._.coref_resolved
    return text
try:
    Doc.set_extension("has_coref", default=False, force=True)
    Doc.set_extension("coref_resolved", default=None, force=True)
except ValueError:
    pass

def find_most_relevant_sentence(question, article_text):
    text = resolve_coreferences(article_text)
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    question_embedding = model.encode(question)
    sentence_embeddings = model.encode(sentences)

    similarities = util.pytorch_cos_sim(question_embedding, sentence_embeddings).squeeze()
    most_similar_index = similarities.argmax().item()
    confidence = similarities[most_similar_index].item()

    return sentences[most_similar_index], confidence

def extract_relevant_snippets(question, relevant_sentence):
    doc = nlp(relevant_sentence)
    question_doc = nlp(question)

    target_label = 'PERSON'  # Default to PERSON

    if any(word in question.lower() for word in ['who', 'name']):
        target_label = 'PERSON'
    elif any(word in question.lower() for word in ['when', 'date', 'year', 'time']):
        target_label = 'DATE'
    elif any(word in question.lower() for word in ['where', 'city', 'country', 'place', 'location']):
        target_label = 'GPE'
    elif any(word in question.lower() for word in ['what', 'company', 'organization']):
        target_label = 'ORG'
    elif 'how many' in question.lower():
        target_label = 'CARDINAL'

    entities = {}
    for ent in doc.ents:
        if ent.label_ == target_label:
            if ent.text in entities:
                entities[ent.text] += 1
            else:
                entities[ent.text] = 1

    if entities:
        sorted_entities = sorted(entities.items(), key=lambda item: (-item[1], relevant_sentence.index(item[0])))
        return sorted_entities[0][0]

    return "No relevant information found."

def answer_question_from_article(article_id, question, df):
    try:
        article_text = df.loc[df['id'] == article_id, 'article'].values[0]
    except IndexError:
        return "Article not found."

    relevant_sentence, confidence = find_most_relevant_sentence(question, article_text)

    confidence_threshold = 0.5
    if confidence < confidence_threshold:
        return "High confidence answer not found."

    answer_snippet = extract_relevant_snippets(question, relevant_sentence)
    return answer_snippet, confidence


def main():
    st.title("Question Answering Systems")
    question = st.text_input("Enter your question:")
    article_id = st.number_input("Enter the article ID:", value=0, step=1)
    
    # Check if the article ID exists in the DataFrame
    if article_id in df['id'].values:

        article_text = df.loc[df['id'] == article_id, 'article'].values[0]
        most_relevant_sentence, _ = find_most_relevant_sentence(question, article_text)

        if st.button("Get Answer"):
            st.write("Most relevant sentence:", most_relevant_sentence)
            answer = answer_question_from_article(article_id, question, df)
            st.write("Answer:", answer)
    else:
        st.write("Article ID not found.")
if __name__ == "__main__":
    main()