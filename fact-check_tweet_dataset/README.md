# Fact-Checks Across Languages
This repository contains data for our paper: Matching Tweets With Applicable Fact-Checks Across Languages by Ashkan Kazemi, Zehua Li, Verónica Pérez-Rosas, Scott A. Hale, and Rada Mihalcea.

### Content:
- `articles.json`: All the fact-checking articles we crawled. We extracted the tweet links from these articles to form `(tweet, fact-check)` pairs, under the assumption that the fact-check articles pertain to the tweets they mention.
- `en_en.csv`: Pairs of English tweets and English articles. For more details regarding the negative example generation process, please check out the “Matching (tweet, fact-check) Pairs” section in our paper. 
- `es_es.csv`: Pairs of Spanish tweets and Spanish articles.
- `pt_pt.csv`: Pairs of Portuguese tweets and Portuguese articles.
- `hi_en.csv`: Pairs of Hindi tweets and English articles.
