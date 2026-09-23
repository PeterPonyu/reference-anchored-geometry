#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build_independent
export TEXINPUTS="build_independent:.:"
xelatex -interaction=nonstopmode -halt-on-error -recorder -output-directory=build_independent paper.tex > build_independent/pass1.log
bibtex build_independent/paper > build_independent/bibtex.log
xelatex -interaction=nonstopmode -halt-on-error -recorder -output-directory=build_independent paper.tex > build_independent/pass2.log
xelatex -interaction=nonstopmode -halt-on-error -recorder -output-directory=build_independent paper.tex > build_independent/pass3.log
cp build_independent/paper.pdf paper.pdf
