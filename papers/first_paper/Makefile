all: paper presentation

paper:
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex

presentation:
	pdflatex slides/main_presentation.tex
	bibtex slides/main_presentation
	pdflatex slides/main_presentation.tex
	pdflatex slides/main_presentation.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc
