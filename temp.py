import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import socket

from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import pdb;

from tqdm._tqdm_notebook import tqdm_notebook

tqdm_notebook.pandas()
from io import BytesIO
from PIL import Image
import urllib
import pytesseract
from urllib.parse import urlparse
from http.client import HTTPConnection, HTTPSConnection
from spacytextblob.spacytextblob import SpacyTextBlob
import re
from pathlib import Path
import spacy
import urllib.request
import cairosvg
from unicodedata import normalize
import cv2

from transformers import AutoTokenizer, AutoModelWithLMHead

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

nlp.add_pipe('spacytextblob')


def clean_text(text):
    text = normalize("NFKD", text)  # Normalization

    text = re.sub(r"[^\w\s]", "", text)  # Remove Punc

    # text = " ".join([token.lemma_ for token in nlp(text) if not token.is_stop])

    text = re.sub("\s+", " ", text)

    text = text.strip()

    return text


def clean_img(img):
    img = cv2.resize(img, (300, 300))

    # Normalization
    img = img / 255.0

    return img


tokenizer = AutoTokenizer.from_pretrained('T5-base')
model = AutoModelWithLMHead.from_pretrained('T5-base', return_dict=True)


def summarize_webpage(sequence):
    inputs = tokenizer.encode("sumarize: " + sequence, return_tensors='pt', max_length=512, truncation=True)
    output = model.generate(inputs, min_length=80, max_length=100)

    summary = tokenizer.decode(output[0])

    return summary


def get_data(image):
    txt = pytesseract.image_to_string(image, lang="eng")
    txt = re.sub("[\n]{2,}", "\t\t", txt)
    txt = re.sub("\n", "", txt)
    txt = re.sub("\t\t", "\n", txt)

    if not txt:
        txt = "No Information"

    return txt


def cos_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)

    return doc1.similarity(doc2)


def sentiment(text):
    doc = nlp(text)

    return doc._.blob.polarity


def check_https_url(url):
    HTTPS_URL = f'https://{url}'
    try:
        HTTPS_URL = urlparse(HTTPS_URL)
        connection = HTTPSConnection(HTTPS_URL.netloc, timeout=2)
        connection.request('HEAD', HTTPS_URL.path)
        if connection.getresponse():
            return 1
        else:
            return 0
    except:
        return 0


def get_host(url):
    res = re.findall("^(www.|https://|http://)", url)
    if res:
        url = re.sub(f"^{res[0]}", "", url)
    try:
        socket.gethostbyname(url)
        return 1
    except:
        return 0


def is_active(url):
    url = url if url[:8] in ["http://", "https://"] else "http://" + url
    try:
        r = requests.head(url, timeout=3)

        if r.status_code == 200:
            return 1
        else:
            return 0
    except:
        return 0


def check_redirect(url):
    try:
        url = url if url[:8] in ["http://", "https://"] else "http://" + url
        r = requests.get(url, timeout=3)
        return len(r.history)
    except:
        return 0

# Example HTML link
html_link = "https://www.naaptol.com"

# Fetch HTML content
response = requests.get(html_link)
html_content = response.text

# Parse HTML content
soup = BeautifulSoup(html_content, "html.parser")



imgs = []
img_txts = []
y_true = []
cos_sims = []
sent = 0
text_from_soup = soup.get_text()


text = summarize_webpage(str(soup))
print(text)
# find all images in URL
images = soup.findAll('img', alt=True)
# print(images)
# checking if images is not zero
if len(images) != 0:
    for i, image in enumerate(images):
        try:
            image_link = image["data-srcset"]
            print("Data 1", image_link)
        except:
            try:
                image_link = image["data-src"]
                print("Data 2", image_link)


            except:
                try:

                    image_link = image["data-fallback-src"]
                except:
                    try:
                        # print("3")
                        image_link = image["src"]

                    except Exception as e:
                        print(f"Error: {e}")

        try:
            if image_link.startswith("/"):
                image_link = "https:" + image_link

            with urllib.request.urlopen(image_link) as response:
                img1 = Image.open(response)

            alt_text = image["alt"]




            if Path(image_link).suffix == ".svg":
                img_png = cairosvg.svg2png(url=image_link)
                img = plt.imread(BytesIO(img_png))[:, :, :3]
                img = np.array(Image.fromarray((img * 255).astype(np.uint8)))

            else:
                img = np.array(img1)
                print(4, type(img))

            sent = sentiment(text)
            img_txt = get_data(img)
            img = clean_img(img)
            cos_sim = cos_similarity(text, img_txt)
            print("we reach here")
            # sent = sentiment(text)
            print(sent)
            # read = readability(text)

            imgs.append(img)
            img_txts.append(img_txt)
            cos_sims.append(cos_sim)
            y_true.append(y)

        except Exception as e:
            print(f"Error: {e}")

print(imgs, img_txts, cos_sims, y_true, sent)



# Further processing and analysis of outputs
# You can perform analysis based on the outputs generated by your functions
