import requests
import time
import os
import logging
import json
import argparse

def get_comments(subreddit, num_comments, api_delay=1):
    base_url = "https://api.pushshift.io/reddit/comment/search"
    comments = []
    counter = 0
    before = None

    logging.info(f'Getting {num_comments} comments from {subreddit}')
    while len(comments) < num_comments:
        if counter % 10 == 0:
            logging.info(f'Got {len(comments) / num_comments * 100:.2f}% so far')
        params = {
            "subreddit": subreddit,
            "size": min(100, num_comments - len(comments)),
            "before": before
        }

        response = requests.get(base_url, params=params)
        data = response.json()["data"]

        if not data:
            break

        comments.extend([comment["body"] for comment in data])
        before = data[-1]["created_utc"]
        time.sleep(api_delay)
        counter+=1
    return comments[:num_comments]


def main(N, debug):
    if debug:
        N = 1

    log_file = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w', format='%(asctime)s %(message)s')
    logging.info(f'Starting script and getting {N} comments per subreddit')

    subs = ['liberal', 'conservative']
    comments_by_sub = {}

    for sub in subs:
        comments_by_sub[sub] = get_comments(sub, N)

    with open('../../data/raw_comments.json', 'w') as outfile:
        json.dump(comments_by_sub, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect comments from Reddit.')
    parser.add_argument('N', metavar='N', type=int, help='Number of comments to fetch per subreddit')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (overrides N to 1)')
    args = parser.parse_args()
    main(args.N, args.debug)
