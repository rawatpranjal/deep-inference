.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib sections/*.tex appendix/*.tex
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf main.toc
