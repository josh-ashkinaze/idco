"""
Author: Joshua Ashkinaze
Date: 03/17/2023

Description: This script preprocesses the comments from the raw_comments.json file. It removes URLs, emojis, special characters, digits, and punctuation. It also converts the text to lowercase, tokenizes the comments, removes stop words, and lemmatizes the comments. It filters out comments with less than 5 words. It outputs the preprocessed comments to processed_comments.json.

"""

import json
import re
import nltk
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import logging
# from spellchecker import SpellChecker

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def clean_unicode(text):
    return text.encode('utf-8').decode('unicode_escape')


def remove_emojis(text):
    return emoji.get_emoji_regexp().sub(r'', text)


# def correct_spelling(tokens):
#     spell = SpellChecker()
#     corrected_tokens = [spell.correction(token) for token in tokens]
#     return corrected_tokens


def preprocess_comment(comment):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    try:
        # Clean Unicode sequences
        cleaned_comment = clean_unicode(comment)

        # Remove URLs
        cleaned_comment = re.sub(r'http\S+', '', cleaned_comment)

        # Remove emojis
        cleaned_comment = remove_emojis(cleaned_comment)

        # Remove special characters, digits, and punctuation
        cleaned_comment = re.sub(r'[^a-zA-Z\s]', '', cleaned_comment)

        # Convert text to lowercase
        cleaned_comment = cleaned_comment.lower()

        # Tokenize comments
        tokens = word_tokenize(cleaned_comment)

        # Remove stop words
        tokens = [token for token in tokens if token not in stop_words]

        # # Lemmatization
        # tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Filter comments with more than 5 words
        if len(tokens) > 5:
            return tokens
        else:
            return None
    except Exception as e:
        logging.info('Error preprocessing comment: {}'.format(e))
        return None

def main():
    log_file = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w', format='%(asctime)s %(message)s')
    logging.info('Starting preprocessing')
    with open('../../data/raw_comments.json', 'r') as infile:
        comments_by_sub = json.load(infile)

    num_subreddits = len(comments_by_sub)
    sub_counter = 0
    comment_counter = 0

    preprocessed_comments_by_sub = {}
    for sub in comments_by_sub:
        logging.info('Preprocessing comments for {}, which is {} of {} subreddits'.format(sub, sub_counter, num_subreddits))
        preprocessed_comments_by_sub[sub] = []
        num_sub_comments = len(comments_by_sub[sub])
        every_10 = num_sub_comments // 10
        for comment in comments_by_sub[sub]:
            if comment_counter % every_10 == 0:
                logging.info('Preprocessing comment {}, which is {} of {} comments'.format(comment, comment_counter/num_sub_comments, num_sub_comments))
            preprocessed_comment = preprocess_comment(comment)
            if preprocessed_comment:
                preprocessed_comments_by_sub[sub].append(preprocessed_comment)
            comment_counter += 1
        sub_counter += 1
    logging.info('Finished preprocessing')
    with open('../../data/processed_comments.json', 'w') as outfile:
        json.dump(preprocessed_comments_by_sub, outfile)


if __name__ == "__main__":
    main()
