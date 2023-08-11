import os
import ssl

import arxiv
from PyPDF2 import PdfReader

from .clean_text import clean_paper_text, slugify

ssl._create_default_https_context = ssl._create_unverified_context


def extract_text(pdf_path: str) -> str:
    """

    :param pdf_path: path to .pdf file
    :return: extracted plain text without preprocessing
    """
    reader = PdfReader(pdf_path)
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        pages.append(text)

    return "\n".join(pages)


def save_pdf(result: arxiv.arxiv.Result, pdf_dir: str) -> None:
    clean_title = slugify(result.title)
    pdf_path = os.path.join(pdf_dir, clean_title + ".pdf")
    result.download_pdf(filename=pdf_path)


def save_extraction(pdf_path: str, txt_path: str) -> None:
    text = extract_text(pdf_path)
    with open(txt_path, "w") as f:
        f.write(clean_paper_text(text))
