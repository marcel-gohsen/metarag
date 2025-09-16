import os.path

import requests
from pybtex.database.input import bibtex


def main():
    parser = bibtex.Parser()
    bib_data = parser.parse_file("data/dagstuhl.bib")

    paper_data = []

    for key, data in bib_data.entries.items():
        try:
            _id = f"https://doi.org/{data.fields['doi']}"

            response = requests.get(f"https://api.openalex.org/works/{_id}")
            response_data = response.json()

            name = os.path.basename(response_data["id"])

            if not os.path.exists(f"/data/paper/md/{name}.md"):
                print(name)
        except KeyError:
            print(data.fields["title"])
        except Exception as e:
            print(data.fields["title"])




if __name__ == '__main__':
    main()
