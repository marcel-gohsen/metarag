import json
import logging
import os
import urllib
import requests
from urllib import request
from urllib.error import HTTPError

import pandas as pd
from retry import retry
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
retry_logger = logging.getLogger("retry")

@retry(tries=5, delay=1, backoff=2, logger=retry_logger)
def download(urls, path):
    try:
        exc = None
        for url in urls:
            try:
               name, headers = request.urlretrieve(url, path)
               if headers["Content-Type"] not in ["application/pdf", "application/x-pdf"]:
                   os.remove(path)
                   raise ValueError("Not a PDF")
            except Exception as e:
                exc = e

        if exc is not None:
            raise exc
    except HTTPError as e:
        # Not recoverable
        if e.code not in [404, 403]:
            raise e

        return e
    except Exception as e:
        return e

    return None

def main():
    logger = logging.getLogger(__name__)

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (X11; Linux x86_64; rv:10.0) Gecko/20100101 Firefox/10.0')]
    urllib.request.install_opener(opener)

    paper_meta_path = "data/openalex-search-result-retrieval-augmented-generation_2025-09-03.csv"
    paper_fulltext_dir = "data/paper/pdf"

    df = pd.read_csv(paper_meta_path)

    stats = {"success": [], "errors": {}, "no_url": []}
    try:
        for _, row in tqdm(df.iterrows()):
            out_path = os.path.basename(row["id"])
            out_path = os.path.join(paper_fulltext_dir, f"{out_path}.pdf")

            if os.path.exists(out_path):
                stats["success"].append(row["id"])
                continue

            urls = set()
            if pd.notna(row["best_oa_location.pdf_url"]):
                urls.add(row["best_oa_location.pdf_url"])
            if pd.notna(row["primary_location.pdf_url"]):
                urls.add(row["primary_location.pdf_url"])
            if pd.notna(row["doi"]):
                urls.add(row["doi"])
                urls.add(f"https://sci-hub.se/{str(row['doi']).replace('https://doi.org/','')}")

            if pd.notna(row["open_access.oa_url"]):
                urls.add(row["open_access.oa_url"])


            source = str(row["primary_location.source.display_name"])
            if "arxiv" in source.lower():
                urls.add(str(row["open_access.oa_url"]).replace("/abs/", "/pdf/"))

            if "acl" in source.lower():
                doi_url = row["doi"]
                if pd.notna(doi_url):
                    x = requests.head(doi_url)


                    if "location" in x.headers and "aclanthology.org" in x.headers["location"]:
                        urls.add(f"{x.headers['location']}.pdf")

            try:
                if len(urls) == 0:
                    stats["no_url"].append(row["id"])
                    continue

                e = download(urls, out_path)

                if e is not None:
                    raise e

                stats["success"].append(row["id"])
            except Exception as e:
                if row["id"] not in stats["errors"]:
                    stats["errors"][row["id"]] = str(e)
    finally:
        with open("crawling_status.json", "w") as f:
            json.dump(stats, f, indent=4)

        logger.info(f"Success: {len(stats['success'])} Errors: {len(stats['errors'])} No_url: {len(stats['no_url'])}")


if __name__ == '__main__':
    main()
