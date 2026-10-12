#!/usr/bin/env bash
# Compile paper.pdf with XeLaTeX and BibTeX. Passes repeat until cross-references settle;
# build_independent/pass3.log holds the log of the final pass.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build_independent
export TEXINPUTS="build_independent:.:"
latex_pass() {
  xelatex -interaction=nonstopmode -halt-on-error -recorder -output-directory=build_independent paper.tex > "build_independent/$1.log"
}
latex_pass pass1
bibtex build_independent/paper > build_independent/bibtex.log
latex_pass pass2
for n in 3 4 5; do
  latex_pass "pass$n"
  if ! grep -q -E "Rerun to get|Label\(s\) may have changed" "build_independent/pass$n.log"; then
    break
  fi
done
[ "$n" = "3" ] || cp "build_independent/pass$n.log" build_independent/pass3.log
cp build_independent/paper.pdf paper.pdf
