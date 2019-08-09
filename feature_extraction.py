from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

test = [["good", "movie"], ["not", "a", "good", "movie"], ["did", "not", "like"], ["i", "like", "it"], ["good", "one"]]

def compute_tfidf(document):
    tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,3))

    lines = []
    for text in document:
        text = " ".join(text)
        lines.append(text)

    features = tfidf.fit_transform(lines)
    df = pd.DataFrame(features.toarray(), columns=tfidf.get_feature_names())

    return df.values


if __name__ == "__main__":
    compute_tfidf(test)