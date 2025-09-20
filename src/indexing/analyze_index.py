import os

from config import CONFIG
from indexing.search_index import ElasticsearchIndex


def main():
    index = ElasticsearchIndex(**CONFIG["index"]["elastic"])

    client = index.es_client

    results = client.search(index=index.index_name, body={"query": {"match_all": {}}, "_source": {"include": ["meta"]}}, size=1000, scroll="2m")
    sid = results["_scroll_id"]

    data = []

    _ids = set()
    while True:
        if len(results["hits"]["hits"]) == 0:
            break

        for hit in results["hits"]["hits"]:
            url = ":".join(hit["_source"]["meta"]["id"].split(":")[0:2])
            _ids.add(os.path.basename(url))
            results.append(hit["_source"])

        results = client.scroll(scroll_id=sid, scroll="2m")

    for paper in os.listdir("data/paper/md"):
        name = os.path.basename(paper).replace(".md", "")
        if name not in _ids:
            print(name)



    index.close()



if __name__ == '__main__':
    main()
