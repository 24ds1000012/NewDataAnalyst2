import requests
import pandas as pd
from bs4 import BeautifulSoup

def scrape_and_parse(url: str, table_index: int = 0):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = pd.read_html(str(soup))
    print (f"Found {len(tables)} tables on the page.")
    return tables[table_index]  # assume first table unless specified