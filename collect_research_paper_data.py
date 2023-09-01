import arxiv

paper_id = '2307.09288'

search = arxiv.Search(id_list=[paper_id])
paper = next(search.results())
print(paper.title)


# Download the PDF to a specified directory with a custom filename.
paper.download_pdf(dirpath="data/", filename=f"{paper.title}.pdf")