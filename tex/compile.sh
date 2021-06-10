#!/bin/bash

pdflatex geosolve.tex
bibtex geosolve.aux
pdflatex geosolve.tex
pdflatex geosolve.tex
