import requests
from bs4 import BeautifulSoup

res = requests.get('https://automatetheboringstuff.com/chapter7/')
soup = BeautifulSoup(res.text, 'html.parser')
print(soup.find('div', { "class" : "book" }).text)