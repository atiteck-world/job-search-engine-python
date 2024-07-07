from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker
import re

app = Flask(__name__)

# Load jobs data
jobs_df = pd.read_csv('data/job_data.csv')
jobs_df.fillna('', inplace=True)

# Combine relevant text fields into a single field for TF-IDF vectorization
jobs_df['combined_text'] = jobs_df['TITLE'] + ' ' + jobs_df['DESCRIPTION'] + ' ' + jobs_df['EMPLOYER_DESCRIPTION'] + ' ' + jobs_df['OCCUPATION']

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(jobs_df['combined_text'])

# Extract unique words from job data to use in spellchecker
def extract_words(text):
    return re.findall(r'\b\w+\b', text.lower())

unique_words = set()
for text in jobs_df['combined_text']:
    unique_words.update(extract_words(text))

# Initialize SpellChecker with custom dictionary
spell = SpellChecker()
spell.word_frequency.load_words(unique_words)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']

        # Spell-checking and suggestions
        query_words = query.split()
        if len(query_words) == 1:
            suggestions = spell.candidates(query_words[0])
            suggestions = sorted(suggestions, key=lambda w: spell.unknown([w]))[:5]
        else:
            suggestions = [query]

        # Collect results for each suggestion
        all_results = []
        for suggestion in suggestions:
            query_vector = vectorizer.transform([suggestion])
            cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
            related_jobs_indices = cosine_similarities.argsort()[::-1][:10]  # Limit to top 10 results
            related_jobs = jobs_df.iloc[related_jobs_indices]
            all_results.extend([
                {
                    'index': index,
                    'job_title': row['TITLE'],
                    'description': row['DESCRIPTION'],
                    'companyDescription': row['EMPLOYER_DESCRIPTION'],
                    'occupation': row['OCCUPATION']
                }
                for index, row in related_jobs.iterrows()
            ])

        # Remove duplicates and keep top 10 results
        seen = set()
        final_results = []
        for job in all_results:
            if job['index'] not in seen:
                final_results.append(job)
                seen.add(job['index'])
                if len(final_results) >= 10:
                    break

        return render_template('index.html', jobs=final_results, query=query, suggestions=suggestions if len(query_words) == 1 else None)
    return render_template('index.html')

@app.route('/job/<int:job_id>')
def job_detail(job_id):
    job = jobs_df.iloc[job_id]
    return render_template('details.html', job={
        'job_title': job['TITLE'],
        'description': job['DESCRIPTION'],
        'companyDescription': job['EMPLOYER_DESCRIPTION'],
        'occupation': job['OCCUPATION'],
        #'company': job['COMPANY'], # Assuming you have this field
        #'location': job['LOCATION'] # Assuming you have this field
    })

if __name__ == '__main__':
    app.run(debug=True)
