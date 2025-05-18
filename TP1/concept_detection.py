import os
import re
import math
import string
import spacy
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake

nlp = spacy.load("fr_core_news_sm")
punct = set(string.punctuation)

# --- Chargement et nettoyage ---

def load_corpus_from_folder(folder_path: str) -> list[str]:
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
    return texts

custom_stopwords = {"du", "de", "la", "le", "des", "et", "à", "en", "un", "une", "dans", 
                    "pour", "que", "qui", "ce", "ces", "cette", "il", "elle", "ils", "elles", 
                    "on", "nous", "vous", "je", "tu", "me", "te", "se", "lui", "leur", "y", "à", 
                    "au", "aux", "par", "avec", "sans", "sous", "sur", "entre", "vers", "depuis", 
                    "avant", "après", "pendant", "comme", "si", "mais", "ou", "donc", "car"}

def get_stopword_set(language="french"):
    stop_words = set(stopwords.words(language))
    try:

        lang_model = spacy.blank(language)
        if hasattr(lang_model, "Defaults") and hasattr(lang_model.Defaults, "stop_words"):
            stop_words.update(lang_model.Defaults.stop_words)
    except:
        pass 
    stop_words.update(custom_stopwords)
    return stop_words

def clean_text(
    text: str,
    lowercase: bool = True,
    remove_stopwords: bool = False,
    output: str = 'text',
    language: str = 'french'
):
    if lowercase:
        text = text.lower()
    if output == 'tokens':
        tokens = word_tokenize(text, language=language)
        if remove_stopwords:
            stop_words = get_stopword_set(language)
            tokens = [t for t in tokens if t not in stop_words]
        return tokens
    elif output == 'text':
        return text
    else:
        raise ValueError("output doit être 'text' ou 'tokens'")



# --- Extraction de ngrammes ---

def extract_ngrams(texts, n, remove_stopwords=False, lowercase=True, language='french'):
    all_ngrams = []
    for text in texts:
        tokens = clean_text(text, lowercase=lowercase, remove_stopwords=remove_stopwords, output='tokens', language=language)
        all_ngrams.extend([' '.join(gram) for gram in ngrams(tokens, n)])
    return all_ngrams


# --- TF-IDF ---

def get_tfidf_keywords(texts, ngram_range=(1,1), max_df=1.0, min_df=1):
    """
    Appliqué à un seul document : max_df=1.0, min_df=1.
    """
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df)
    X = vectorizer.fit_transform(texts)
    scores = X.mean(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    return sorted(zip(vocab, scores), key=lambda x: -x[1])


# --- PMI ---

def get_pmi_keywords(texts, n=2, threshold=2):
    unigram_freq = Counter()
    ngram_freq = Counter()
    total_unigrams = 0
    total_ngrams = 0

    for text in texts:
        tokens = clean_text(text, output='tokens', language="french")
        unigram_freq.update(tokens)
        total_unigrams += len(tokens)
        grams = list(ngrams(tokens, n))
        ngram_freq.update([' '.join(g) for g in grams])
        total_ngrams += len(grams)

    scores = {}
    for ng in ngram_freq:
        count_ng = ngram_freq[ng]
        if count_ng < threshold:
            continue
        parts = ng.split()
        p_ng = count_ng / total_ngrams
        p_parts = math.prod([unigram_freq[w] / total_unigrams for w in parts])
        if p_parts > 0:
            pmi = math.log2(p_ng / p_parts)
            scores[ng] = pmi
    return sorted(scores.items(), key=lambda x: -x[1])

# --- C-value ---

def get_cvalue_keywords(texts, n=2):
    freq = Counter()
    for text in texts:
        tokens = clean_text(text, output='tokens', language="french")
        grams = ngrams(tokens, n)
        freq.update([' '.join(g) for g in grams])
    
    c_values = {}
    for term, freq_term in freq.items():
        containing_terms = [t for t in freq if term in t and t != term]
        if containing_terms:
            nested_freqs = [freq[t] for t in containing_terms]
            adjusted_freq = freq_term - sum(nested_freqs) / len(nested_freqs)
        else:
            adjusted_freq = freq_term
        cval = math.log2(len(term.split()) + 1) * adjusted_freq
        c_values[term] = cval
    return sorted(c_values.items(), key=lambda x: -x[1])

# --- RAKE ---

def get_rake_keywords(texts, language='french'):
    if language == "french":
        from rake_nltk import Rake
        from nltk.corpus import stopwords
        stop_fr = stopwords.words('french')
        rake = Rake(language=None, stopwords=stop_fr)
    else:
        rake = Rake(language=language)

    keywords = []
    for text in texts:
        rake.extract_keywords_from_text(text)
        keywords.extend(rake.get_ranked_phrases_with_scores())
    scores = defaultdict(float)
    for score, phrase in keywords:
        scores[phrase] += score
    return sorted(scores.items(), key=lambda x: -x[1])


# --- Filtrage syntaxique ---

def filter_extracted_ngrams(ngrams_list: list[str], language_model="fr_core_news_sm") -> list[str]:
    nlp = spacy.load(language_model)
    filtered = []
    parsed_ngrams = [(ng, nlp(ng)) for ng in ngrams_list]
    

    for ng, doc in parsed_ngrams:
        tokens = list(doc)

        # Trop court
        if len(tokens) < 1:
            continue

        # Début ou fin = stopword
        if tokens[0].is_stop or tokens[-1].is_stop:
            continue

        # Contient un verbe conjugué ou auxiliaire
        if any(tok.pos_ in {"AUX", "VERB", "ADV"} for tok in tokens):
            continue

        # Entièrement fonctionnel
        if not any(tok.pos_ in {"NOUN", "PROPN"} for tok in tokens):
            continue


        # Trop de stopwords
        stop_ratio = sum(tok.is_stop for tok in tokens) / len(tokens)
        if stop_ratio > 0.5:
            continue

        # Pas de nom
        if not any(tok.pos_ in {"NOUN", "PROPN"} for tok in tokens):
            continue

        # Contient de la ponctuation
        if any(tok.is_punct for tok in tokens):
            continue

        filtered.append(ng)

    # Supprimer les sous-chaînes
    final = []
    for ng in filtered:
        if not any(ng != other and ng in other for other in filtered):
            final.append(ng)

    return final



