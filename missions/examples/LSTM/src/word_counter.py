from collections import Counter
import konlpy
from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Hannanum

text_file = open("../../movie-review/sample_data/movie_review/train/train_data")
corpus = text_file.read()
print(corpus.split("\n"))
vectorizer = CountVectorizer(min_df=0, ngram_range=(1,1))
X = vectorizer.fit_transform(corpus.split("\n"))
Xc = X.T * X
Xc.setdiag(0)

result = Xc.toarray()

print(result[0])