import os
import pdfplumber
import math
from collections import defaultdict, Counter
import argparse
import json
import pickle

STORAGE_FILE = 'pdf_search_data.pkl'

def extract_text_from_pdfs(folder_path):
    print("Extracting text from PDFs...")
    corpus = []
    pdf_file_text = {}
    inverted_index = defaultdict(lambda: defaultdict(list))
    
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        pdf_path = os.path.join(folder_path, pdf_file)
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                text += page_text
                
                words = preprocess_text(page_text).split()
                for word in set(words):
                    inverted_index[word][pdf_file].append(page_num)
                
        corpus.append(text)
        pdf_file_text[pdf_file] = text
    return corpus, pdf_file_text, inverted_index

def preprocess_text(text):
    return ''.join([char.lower() if char.isalnum() or char.isspace() else ' ' for char in text])

def compute_tf(corpus):
    print("Computing TF scores...")
    tf_scores = []
    for document in corpus:
        preprocessed_document = preprocess_text(document)
        term_count = Counter(preprocessed_document.split())
        total_terms = len(preprocessed_document.split())
        tf = {term: count / total_terms for term, count in term_count.items()}
        tf_scores.append(tf)
    return tf_scores

def compute_idf(corpus):
    print("Computing IDF scores...")
    num_documents = len(corpus)
    document_frequencies = defaultdict(int)
    
    for document in corpus:
        preprocessed_document = preprocess_text(document)
        unique_terms = set(preprocessed_document.split())
        for term in unique_terms:
            document_frequencies[term] += 1
    
    idf_scores = {}
    for term, doc_freq in document_frequencies.items():
        idf_scores[term] = math.log(num_documents / (doc_freq + 1)) + 1
    return idf_scores

def compute_tfidf(tf_scores, idf_scores):
    print("Computing TF-IDF scores...")
    tfidf_scores = []
    for tf in tf_scores:
        tfidf = {term: tf_val * idf_scores.get(term, 0) for term, tf_val in tf.items()}
        tfidf_scores.append(tfidf)
    return tfidf_scores

def search_query(query, tfidf_scores, pdf_file_text, inverted_index):
    query_terms = preprocess_text(query).split()
    doc_scores = defaultdict(float)
    doc_pages = defaultdict(set)

    for term in query_terms:
        for doc, pages in inverted_index.get(term, {}).items():
            doc_index = list(pdf_file_text.keys()).index(doc)
            score = tfidf_scores[doc_index].get(term, 0)
            doc_scores[doc] += score
            doc_pages[doc].update(pages)

    ranked_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)

    results = []
    for doc, score in ranked_docs:
        if score > 0:
            pages = sorted(doc_pages[doc])
            results.append({
                'document': doc,
                'score': score,
                'pages': pages
            })

    return results

def save_data(folder_path, corpus, pdf_file_text, inverted_index, tf_scores, idf_scores, tfidf_scores):
    print("Saving data to file...")
    data = {
        'corpus': corpus,
        'pdf_file_text': pdf_file_text,
        'inverted_index': dict(inverted_index),
        'tf_scores': tf_scores,
        'idf_scores': idf_scores,
        'tfidf_scores': tfidf_scores
    }
    with open(os.path.join(folder_path, STORAGE_FILE), 'wb') as f:
        pickle.dump(data, f)

def load_data(folder_path):
    print("Loading data from file...")
    with open(os.path.join(folder_path, STORAGE_FILE), 'rb') as f:
        data = pickle.load(f)
    data['inverted_index'] = defaultdict(lambda: defaultdict(list), data['inverted_index'])
    return data['corpus'], data['pdf_file_text'], data['inverted_index'], data['tf_scores'], data['idf_scores'], data['tfidf_scores']

def main():
    parser = argparse.ArgumentParser(description="Enhanced PDF TF-IDF Search Tool")
    parser.add_argument('folder', type=str, help="Folder path containing PDF files")
    parser.add_argument('--update', action='store_true', help="Update the stored data")
    
    args = parser.parse_args()
    
    storage_file_path = os.path.join(args.folder, STORAGE_FILE)
    
    if os.path.exists(storage_file_path) and not args.update:
        corpus, pdf_file_text, inverted_index, tf_scores, idf_scores, tfidf_scores = load_data(args.folder)
    else:
        corpus, pdf_file_text, inverted_index = extract_text_from_pdfs(args.folder)
        tf_scores = compute_tf(corpus)
        idf_scores = compute_idf(corpus)
        tfidf_scores = compute_tfidf(tf_scores, idf_scores)
        save_data(args.folder, corpus, pdf_file_text, inverted_index, tf_scores, idf_scores, tfidf_scores)
    
    print("Ready to search. Enter your query (or press Enter to exit):")
    while True:
        query = input("Query: ").strip()
        if not query:
            print("Exiting the program. Thank you for using the PDF search tool!")
            break
        
        results = search_query(query, tfidf_scores, pdf_file_text, inverted_index)
        
        if results:
            print("\nSearch results:")
            for result in results:
                print(f"Document: {result['document']} - Score: {result['score']:.4f}")
                print(f"  Found on page(s): {', '.join(map(str, result['pages']))}")
        else:
            print("No results found for the given query.")
        print()

if __name__ == '__main__':
    main()