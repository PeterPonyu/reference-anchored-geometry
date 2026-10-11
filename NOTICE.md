# Attribution and reuse boundaries

The manuscript bibliography attributes the original source studies and methods.
The author licenses only contributions for which the author holds the necessary
rights; public accessibility, a citation or a checksum is not itself a new
permission to redistribute third-party material.

Included model outputs, summaries and synthetic fixtures are distinguished from
source-study expression, cell/gene identifiers, observed labels and existing
computational annotations. Source-derived fields in mixed NPZ or CSV files are
not newly licensed under CC BY 4.0. The source-study terms continue to apply.
Neither a predicted cell state nor a latent coordinate is an independent
biological observation. No raw sequencing files or private clinical records are
claimed as part of this release.

The arrays of the seven tissue sections and the blood sample
(`evidence/followup-20261010/arrays/diversity/`) combine representation
coordinates computed by the author with tissue positions and spot or cell
barcodes taken from the public source datasets cited in the manuscript: the
Ståhl olfactory-bulb section (Science 2016), the LIBD dorsolateral
prefrontal cortex sections 151673 and 151507 (Nature Neuroscience 2021), the
breast-cancer liver-metastasis section A_0069226 (Zenodo,
https://doi.org/10.5281/zenodo.18306028, CC BY 4.0), the 10x Genomics Visium
human lymph node, breast cancer and ovarian cancer sections, and the 10x
Genomics 3k PBMC sample (landing pages are given in the manuscript
bibliography). The positions and barcodes keep the source terms. Source-provided
observed labels, such as cortical-layer annotations, are not redistributed; only
scores computed from them appear. Count matrices of those datasets are not
distributed. `evidence/followup-20261010/upstream/audit/` holds two author-written
Python modules (MIT) that generate the synthetic fixtures and implement
co-ranking quantities.

Where a frozen software snapshot includes an original copyright notice, that
notice is preserved. The MIT grant for the author's contributions does not
replace third-party notices. Standard software dependencies and fonts are
installed separately rather than distributed as standalone font/library files.

Author: Zeyu Fu. Contact: fuzeyu09@gmail.com.
