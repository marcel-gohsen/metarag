import csv
import pandas as pd


def main():
    df = pd.read_csv("data/openalex-search-result-retrieval-augmented-generation_2025-09-03.csv")
    print(df["primary_location.pdf_url"].count())



if __name__ == '__main__':
    main()
