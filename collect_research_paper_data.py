import arxiv

paper_id = '2306.05499'


search = arxiv.Search(id_list=[paper_id])
paper = next(search.results())
print(paper.title)

paper.download_pdf(filename=f"data/{paper.title}.pdf")