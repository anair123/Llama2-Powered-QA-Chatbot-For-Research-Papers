import arxiv

paper_ids = ['2109.02734', '2303.10311', '2003.12309', '1808.02191']

for paper_id in paper_ids:
    search = arxiv.Search(id_list=[paper_id])
    paper = next(search.results())
    print(paper.title)

    paper.download_pdf()