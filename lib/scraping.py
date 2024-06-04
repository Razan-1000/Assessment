# https://u.ae/en/information-and-services#/

from bs4 import BeautifulSoup
import requests

request= requests.get("https://u.ae/en/information-and-services#/")

soup= BeautifulSoup(request.text, "html.parser")


print(soup.prettify())
# Find elements with the specified class
elements = soup.find_all(class_="row ui-filter-items row-flex")

# Print the elements
for element in elements:
    print(element)