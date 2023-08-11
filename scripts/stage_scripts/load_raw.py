import sys
from pathlib import Path

root = str(Path(__file__).parent.parent.parent)
sys.path.append(root)


import os
import click

import arxiv
from dvc.api import params_show

from src.data.parsing import save_pdf


@click.command()
@click.option('--query', type=str, help='arxiv search query')
@click.option('--max_results', type=int, default=1)
def main(query, max_results):

    dvc_params = params_show()
    pdf_dir = dvc_params["paths"]["pdf_dir"]

    os.makedirs(pdf_dir, exist_ok=True)

    search = arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate
    )
    num_processed = 0
    for result in search.results():
        # toDo logging
        print(f"processing paper {num_processed}")
        save_pdf(result, pdf_dir)
        num_processed += 1


if __name__ == "__main__":
    main()
