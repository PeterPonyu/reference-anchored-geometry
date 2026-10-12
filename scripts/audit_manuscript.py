"""Self-contained audit of the manuscript source, compiled PDF, figures and evidence.

Run from the repository root after `bash build.sh`:   python scripts/audit_manuscript.py [--write]

Checks
  structure   Introduction and Discussion are continuous prose; Methods hold equations but no floats; all figures
              and tables sit in Results; panel callouts and caption definitions are complete
  citations   every cited key is defined, the bibliography holds exactly the cited keys, numbering follows first citation
  language    no internal version codes, hashes, file names or process notes in the body text
  floats      every figure and table is called out no later than the page on which it appears; numbering follows
              first citation
  layout      LaTeX log without diagnostics, abstract at most 200 words, no page with more than 1.5 in of empty space
              above the bottom margin, fonts embedded
  figures     one page, expected bold panel tags only, Arial text, nothing outside the page
  evidence    SHA-256 manifest of the saved evidence matches the files
  numbers     values added during the final revision match the saved records and the text
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path

import fitz
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "evidence/submission-20261011/results"
FIGURES = {
    "architecture": "abcd", "calibration_perturbations": "abcdef", "calibration_reachability": "abcdef",
    "coherence_baselines": "abcd", "allocation_validity_guards": "abcd", "inventory_null_comparison": "abcd",
    "scale_reference_sensitivity": "abcd", "row_correspondence_sensitivity": "abcd", "chance_correction": "abcdef",
    "spatial_reference": "abcdefgh", "object_embeddings": "abcdefghij", "object_pbmc": "abcdef",
}
errors: list[str] = []
report: dict = {}


def fail(message: str) -> None:
    errors.append(message)


tex = (ROOT / "paper.tex").read_text()
body = re.sub(r"(?<!\\)%.*", "", tex)
pos = {name: body.index(marker) for name, marker in (
    ("intro", "\\section{Introduction}"), ("methods", "\\section{Materials and Methods}"),
    ("results", "\\section{Results}"), ("discussion", "\\section{Discussion}"),
    ("declarations", "\\section{Declarations}"), ("end", "\\bibliographystyle"))}
section = {"intro": body[pos["intro"]:pos["methods"]], "methods": body[pos["methods"]:pos["results"]],
           "results": body[pos["results"]:pos["discussion"]], "discussion": body[pos["discussion"]:pos["declarations"]],
           "declarations": body[pos["declarations"]:pos["end"]]}

# ------------------------------------------------------------------ structure
for name in ("intro", "discussion"):
    if re.search(r"\\subsection|\\begin\{(figure|table|itemize|enumerate|equation|align)\}|\\item|\\figpanel|\\ref\{(fig|tab):",
                 section[name]):
        fail(f"{name} is not continuous prose")
if re.search(r"\\begin\{(figure|table)\}", section["methods"]):
    fail("Methods contain a figure or table")
floats = list(re.finditer(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", body, re.S))
counts = {k: sum(m[1] == k for m in floats) for k in ("figure", "table")}
if counts != {"figure": 12, "table": 4}:
    fail(f"float counts {counts}")
if not all(pos["results"] < m.start() < pos["discussion"] for m in floats):
    fail("a float lies outside Results")
equation_blocks = [m.start() for m in re.finditer(r"\\begin\{(?:equation|align)\*?\}|\\\[", body)]
if not all(pos["methods"] < p < pos["results"] for p in equation_blocks):
    fail("a displayed equation lies outside Methods")
equation_labels = re.findall(r"\\label\{(eq:[^}]+)\}", body)
if len(equation_labels) != len(set(equation_labels)):
    fail("duplicate equation labels")
report["structure"] = {"floats": counts, "displayed_equations": len(equation_blocks)}

labels = {}
for m in floats:
    label = re.search(r"\\label\{([^}]+)\}", m[0])[1]
    caption = re.search(r"\\caption\{(.*?)\}\s*\\label", m[0], re.S)[1]
    graphic = re.search(r"\\includegraphics[^{]*\{([^}]+)\}", m[0])
    labels[label] = dict(kind=m[1], start=m.start(), caption=caption,
                         file=Path(graphic[1]).stem if graphic else None)
fig_labels = [k for k, v in labels.items() if v["kind"] == "figure"]
tab_labels = [k for k, v in labels.items() if v["kind"] == "table"]
# panel callouts and caption definitions
if sorted(labels[k]["file"] for k in fig_labels) != sorted(FIGURES):
    fail("figure files in the text differ from the expected twelve")
for label in fig_labels:
    info, name = labels[label], labels[label]["file"]
    tags = set(FIGURES.get(name, ""))
    called = set()
    for m in re.finditer(r"\\figpanel\{%s\}\{([^}]+)\}" % re.escape(label), body):
        spec = m.group(1)
        called |= {chr(c) for c in range(ord(spec[0]), ord(spec[-1]) + 1)} if re.fullmatch(r"[a-z]--[a-z]", spec) else set(spec)
    defined = set(re.findall(r"(?<!\w)([a-z]),", info["caption"]))
    if called != tags:
        fail(f"{label}: panels without callout {sorted(tags - called)} or extra {sorted(called - tags)}")
    if defined != tags:
        fail(f"{label}: caption definitions {sorted(defined)} differ from {sorted(tags)}")
if re.search(r"\\ref\*?\{fig:", body):
    fail("whole-figure callout")

# -------------------------------------------------------------------- citations
cites = []
for m in re.finditer(r"\\cite[a-z]*\{([^}]+)\}", body):
    for key in m.group(1).split(","):
        key = key.strip()
        if key not in cites:
            cites.append(key)
bib_text = (ROOT / "references.bib").read_text()
bib_keys = re.findall(r"@\w+\{([^,\s]+),", bib_text)
if len(bib_keys) != len(set(bib_keys)):
    fail("duplicate bibliography keys")
if set(cites) != set(bib_keys):
    fail(f"bibliography differs from citations: uncited {sorted(set(bib_keys) - set(cites))}, undefined {sorted(set(cites) - set(bib_keys))}")
bbl_keys = re.findall(r"\\bibitem\[[^\]]*(?:\n[^\]]*)*\]\{([^}]+)\}", (ROOT / "build_independent/paper.bbl").read_text())
if bbl_keys != cites:
    fail("printed reference order differs from first-citation order")
report["citations"] = {"cited": len(cites), "bibliography_entries": len(bib_keys), "intro": len(set(re.findall(r"\\cite[a-z]*\{([^}]+)\}", section['intro']))),
                       "discussion_cited": sorted({k.strip() for g in re.findall(r"\\cite[a-z]*\{([^}]+)\}", section['discussion']) for k in g.split(',')})}
if re.search(r"\\cite", section["results"]):
    fail("Results cite the bibliography")

# --------------------------------------------------------------------- language
banned = {
    "version code": r"RAAG v\d|\bv[23]\b|separately versioned",
    "hash or digest": r"SHA-?256|digest|checksum|\bhash",
    "file name": r"\bNPZ\b|\bnpz\b|README|\.(py|csv|json|npz)\b",
    "process language": r"follow-?up|working revision|local working|locked|smoke|not asserted|remain(s)? unresolved|remain(s)? open|pre-submission|before submission|continuation|withdrawal|prospectively registered|revised damage|earlier measurement|original study|at this stage|new tie rule|machine-readable|new sensitivity|new scores|a new fixed",
    "code identifier": r"\\texttt|non_native|near_uniform|softmax_coordinate",
    "placeholder": r"TODO|TBD|XXX",
    "mixed spelling": r"\bgray\b|\bGray\b",
}
visible = re.sub(r"\\includegraphics[^{]*\{[^}]*\}", "", body[pos["intro"]:pos["end"]])
visible = re.sub(r"\\input\{[^}]*\}", "", visible)
language = {}
for kind, pattern in banned.items():
    hits = [visible[max(0, m.start() - 40):m.end() + 40].replace("\n", " ") for m in re.finditer(pattern, visible)]
    language[kind] = len(hits)
    if hits:
        fail(f"{kind}: {hits[:3]}")
report["language"] = language

# -------------------------------------------------------------------------- PDF
pdf_path = ROOT / "build_independent/paper.pdf"
doc = fitz.open(pdf_path)
report["pages"] = len(doc)
log = (ROOT / "build_independent/pass3.log").read_text(errors="replace")
diagnostics = [ln for ln in log.splitlines() if re.search(r"Overfull|Underfull|LaTeX Warning|Package .* Warning|Missing character|undefined", ln, re.I)]
if diagnostics:
    fail(f"LaTeX diagnostics: {diagnostics[:3]}")
page1 = doc[0].get_text()
abstract = re.sub(r"-\n\s*", "", page1.split("Abstract", 1)[1].split("1\nIntroduction", 1)[0])
report["abstract_words"] = len(abstract.split())
if report["abstract_words"] > 200:
    fail("abstract exceeds the journal 200-word limit")
title = re.search(r"pdftitle=\{([^}]+)\}", tex)[1]
if doc.metadata.get("title") != title:
    fail("PDF title metadata")
fonts = subprocess.run(["pdffonts", str(pdf_path)], capture_output=True, text=True).stdout.splitlines()[2:]
if any(len(f.split()) >= 5 and f.split()[-5] != "yes" for f in fonts):
    fail("a font is not embedded")
gaps = []
for i, page in enumerate(doc, 1):
    pix = page.get_pixmap(dpi=30, colorspace=fitz.csGRAY)
    ink = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w) < 170
    limit = int(pix.h - 1.0 * 30)
    rows = np.where(ink[:limit].any(axis=1))[0]
    gaps.append(round((limit - 1 - rows[-1]) / 30, 2))
# the last page ends the reference list wherever it falls; it only has to be more than a stub
report["largest_empty_space_above_bottom_margin_in"] = max(gaps[:-1])
report["pages_with_more_than_1in_empty"] = [i + 1 for i, g in enumerate(gaps[:-1]) if g > 1.0]
report["last_page_filled_in"] = round(9.0 - gaps[-1], 2)
if max(gaps[:-1]) > 1.5:
    fail(f"page with {max(gaps[:-1])} in of empty space: {[i + 1 for i, g in enumerate(gaps[:-1]) if g > 1.5]}")
if report["last_page_filled_in"] < 2.5:
    fail(f"the last page holds only {report['last_page_filled_in']} in of content")

# ----------------------------------------------------------------- page flow
# body text is 10.9 pt, captions 10.0 pt; the footer lies below y = 722 pt
flow = []
for page in doc:
    body, heads, caps = [], [], []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        for line in b["lines"]:
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text:
                continue
            span, (y0, y1) = line["spans"][0], (line["bbox"][1], line["bbox"][3])
            if re.match(r"^(Figure|Table) \d+:", text) and abs(span["size"] - 10.0) < 0.3:
                caps.append(y0)
            elif 10.8 <= span["size"] <= 11.0 and y1 <= 722 and not span["flags"] & 16:
                body.append((y0, y1, text))
            elif span["flags"] & 16 and span["size"] >= 10.9 and re.match(r"^\d+(\.\d+)*\s", text) and y1 <= 722:
                heads.append((y0, y1))
    flow.append((sorted(body), heads, caps))
hyphen_breaks, stranded, split_by_float = [], [], []
for i, (body, heads, caps) in enumerate(flow):
    if i < len(flow) - 1 and body and body[-1][2].endswith("-"):
        hyphen_breaks.append(i + 1)
    stranded += [i + 1 for (y0, y1) in heads if sum(b[0] > y1 for b in body) < 3]
    if i > 0 and body and body[0][2][:1].islower() and (any(c < body[0][0] for c in caps) or not flow[i - 1][0]):
        split_by_float.append(i + 1)
report["page_flow"] = {"hyphen_at_page_end": hyphen_breaks, "stranded_headings": stranded, "sentence_split_by_float": split_by_float}
if hyphen_breaks:
    fail(f"a page ends on a hyphenated word: pages {hyphen_breaks}")
if stranded:
    fail(f"a heading is stranded at a page bottom: pages {stranded}")
if split_by_float:
    fail(f"a sentence is interrupted by a float: pages {split_by_float}")

# ------------------------------------------------------------- float call-outs
page_text = [p.get_text() for p in doc]
caption_page, first_call = {}, {}
numbering = {"figure": {}, "table": {}}
order = {"figure": fig_labels, "table": tab_labels}
for kind, items in order.items():
    for n, label in enumerate(items, 1):
        word = "Figure" if kind == "figure" else "Table"
        cap = next((i for i, t in enumerate(page_text) if re.search(rf"(?m)^{word} {n}:", t)), None)
        call = next((i for i, t in enumerate(page_text)
                     if re.search(rf"{word} {n}(?![0-9])(?!:)", re.sub(rf"(?m)^{word} {n}:.*", "", t))), None)
        caption_page[label], first_call[label] = cap, call
        if cap is None or call is None:
            fail(f"{label}: caption or call-out not found in the PDF")
        elif call > cap:
            fail(f"{label}: first called out on page {call + 1}, appears on page {cap + 1}")
    calls = [first_call[l] for l in items]
    if None not in calls and calls != sorted(calls):
        fail(f"{kind} numbering does not follow first call-out: {calls}")
report["float_pages"] = {l: (caption_page[l] + 1 if caption_page[l] is not None else None) for l in fig_labels + tab_labels}

# ---------------------------------------------------------------------- figures
for name, tags in FIGURES.items():
    fdoc = fitz.open(ROOT / "figures" / (name + ".pdf"))
    spans = [s for b in fdoc[0].get_text("dict")["blocks"] if "lines" in b for l in b["lines"] for s in l["spans"] if s["text"].strip()]
    bold = [s for s in spans if s["flags"] & 16 or "bold" in s["font"].lower()]
    actual = "".join(sorted(s["text"].strip() for s in bold))
    outside = [s["text"] for s in spans if not fdoc[0].rect.contains(fitz.Rect(s["bbox"]))]
    italic = [s["text"] for s in spans if s["flags"] & 2 or "italic" in s["font"].lower()]
    bad_fonts = sorted({s["font"] for s in spans if "arial" not in s["font"].lower()})
    if len(fdoc) != 1 or actual != tags or outside or italic or bad_fonts:
        fail(f"figure {name}: tags {actual!r} outside {outside} italic {italic} fonts {bad_fonts}")

# --------------------------------------------------------------------- evidence
manifest = ROOT / "SHA256SUMS"
if manifest.exists():
    bad = []
    for line in manifest.read_text().splitlines():
        digest, _, rel = line.partition("  ")
        target = ROOT / rel
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            bad.append(rel)
    report["evidence_manifest_entries"] = len(manifest.read_text().splitlines())
    # rebuilt PDFs carry build timestamps, so a rebuild changes their bytes without changing their content
    rebuilt = [name for name in bad if name == "paper.pdf" or re.fullmatch(r"figures/[^/]+\.pdf", name)]
    bad = [name for name in bad if name not in rebuilt]
    report["rebuilt_pdfs_not_byte_identical"] = rebuilt
    if bad:
        fail(f"{len(bad)} files differ from SHA256SUMS: {bad[:5]}")
else:
    fail("SHA256SUMS is missing")

# ---------------------------------------------------------------------- numbers
ties = json.loads((RESULT / "tie_claims.json").read_text())
traj = json.loads((RESULT / "trajectory_summary.json").read_text())
def outward(values, places):
    """Minimum rounded down and maximum rounded up to the printed precision, so the printed interval contains every value."""
    q = Decimal(1).scaleb(-places)
    lo = Decimal(str(min(values))).quantize(q, rounding=ROUND_FLOOR)
    hi = Decimal(str(max(values))).quantize(q, rounding=ROUND_CEILING)
    return f"{lo:.{places}f}", f"{hi:.{places}f}"


def rng(values, places):
    lo, hi = outward(values, places)
    return f"{lo}--{hi}"


def bound(value, places):
    """A printed upper bound rounded up so that it is not below the value."""
    return f"{Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_CEILING):.{places}f}"


rng_all = ties["largest_range_over_all_62_graphs"]
shift = ties["largest_tie_complete_shift"]
checks = [
    ("tie E range", rng_all["E"] <= 0.0015, "0.0015"), ("tie E* range", rng_all["E_star"] <= 0.0021, "0.0021"),
    ("tie R range", rng_all["R"] == 0, "reachability not at all"), ("tie S range", rng_all["S"] <= 0.013, "0.013"),
    ("tie S* range", rng_all["S_star"] <= 0.043, "0.043"), ("tie-complete E* shift", shift["E_star"] <= 0.0017, "0.0017"),
    ("tie-complete S* shift", shift["S_star"] <= 0.040, "0.040"),
    ("UMAP E* range", True, rng(ties["umap_Estar_range_all_graphs"], 4) + " for UMAP"),
    ("random E* range", True, rng(ties["random_Estar_range_all_graphs"], 4) + " for the random projection"),
    ("orderings hold in every graph", all(ties["held_in_every_graph"].values()), "Every ordering reported above held in every graph"),
    ("saved-order scores reproduced", all(v == 0 for v in ties["reproduction_max_abs_difference_vs_saved"].values()), "reproduce the stored scores exactly"),
    ("identity root", traj["identity"]["max_abs_change_with_true_root"] <= 0.0005 and max(traj["identity"]["root_known_pseudotime"]) <= 0.0067, "0.06 or less"),
    ("reversed-case root", traj["arms_reversed_root"]["max_root_known_pseudotime"] <= 0.0601 and traj["arms_reversed_root"]["max_abs_change_with_true_root"] <= 0.010, "0.010"),
    ("reversed cases", traj["arms_reversed_root"]["n_cases"] == 7, "seven cases in all"),
    ("trunk range", round(traj["arms_reversed_root"]["trunk_rho_range"][0], 3) == 0.977 and round(traj["arms_reversed_root"]["trunk_rho_range"][1], 3) == 0.996, "0.977--0.996"),
    ("arm range", round(traj["arms_reversed_root"]["arm_rho_range"][0], 3) == -0.907 and round(traj["arms_reversed_root"]["arm_rho_range"][1], 3) == -0.852, "$-0.852$ to $-0.907$"),
    ("jitter draws", traj["jitter_restored_draws"] == {"0.002": [0, 15], "0.006": [0, 15], "0.01": [5, 15]}, "five of 15 draws"),
    ("three diffusion components", (round(min(traj["probe_ranges"]["diffusion_components|3.0"]), 4), round(max(traj["probe_ranges"]["diffusion_components|3.0"]), 4)) == (0.9995, 0.9997), "0.9995--0.9997 in all three seeds"),
    ("30 neighbours", (round(min(traj["probe_ranges"]["neighbours|30.0"]), 4), round(max(traj["probe_ranges"]["neighbours|30.0"]), 4)) == (0.9986, 0.9995), "0.9986--0.9995"),
    ("shatter components", all(r["n_components"] == 3 and abs(r["largest_component_fraction"] - 1 / 3) < 1e-3 for r in traj["shatter"]), "three components of one third"),
    ("reproduction", traj["reproduction_max_abs_difference"] < 1e-4, "scores only 0.383--0.391"),
]
calibration = json.loads((ROOT / "evidence/measurement-inventory/calibration.json").read_text())
official = {}
for row in calibration["rows"]:
    official.setdefault(row["candidate"], []).append(row["stacks"]["scib_official"]["values"]["scib_trajectory_conservation"])


def span(names, places=3):
    return rng([v for name in names for v in official[name]], places)


checks += [
    ("official faithful range", True, span(["faithful"]) + " for faithful"),
    ("official matched shatter range", True, span(["shatter_0.5", "shatter_2.5", "shatter_9"]) + " for matched shatter"),
    ("official complete collapse range", True, "complete collapse scores " + span(["collapse_0"])),
    ("official identity range", True, "scores only " + span(["identity"])),
    ("official partial collapse values", True, "(" + ", ".join(f"{v:.3f}" for v in official["collapse_0.5"][:2]) + " and " + f"{official['collapse_0.5'][2]:.3f}" + ")"),
]
checks += [("identity probe range", True, "where the 15-neighbour graph gave " + span(["identity"]) + ".")]
follow = ROOT / "evidence/followup-20261010/results"
div = pd.read_csv(follow / "a2_diversity.csv")
tissue = div[div["reference"] == "tissue"]
umap_t, random_t = tissue[tissue["candidate"] == "umap"], tissue[tissue["candidate"] == "random"]
others_t = tissue[tissue["candidate"].isin(["pca", "scvi", "random"])]
perm = pd.read_csv(follow / "a4_native_permutation_unclipped.csv")
alloc = pd.read_csv(follow / "a4_coherence_allocations.csv")
native = alloc[alloc["family"] == "native"]
scale = json.loads((ROOT / "evidence/scale-correspondence/results.json").read_text())["scale"]
by_id = {}
for row in scale:
    if row["native"]:
        by_id.setdefault(row["id"], []).append(row["allocation_coherence_raw"])
c_ranges = [max(v) - min(v) for v in by_id.values()]
sections = json.loads((RESULT / "tie_sections.json").read_text())
fractions = [100 * s["cutoff_tie_fraction"] for s in sections if s["cutoff_tie_fraction"] > 0]
refit = json.loads((follow / "a5_summary.json").read_text())["seed_variability"]
e_star_sd = [max(r["sd_scvi"], r["sd_umap"]) for r in refit if r["metric"] == "tissue_E_star"]
bulb = [r for r in refit if r["object"] == "bulb" and r["metric"] == "tissue_E_star"][0]
probe = json.loads((ROOT / "evidence/robustness-audit/tied-graph-probe.json").read_text())["joint_reorder"]
d_e = max(abs(r["changes"].get("edge_survival", 0)) for r in probe)
d_s = max(abs(r["changes"].get("separation_survival", 0)) for r in probe)
lo, hi = outward(umap_t["E_star"], 3)
lo_r, hi_r = outward(random_t["E_star"], 3)
checks += [
    ("saved-order UMAP E*", True, f"($E^*$ between {lo} and {hi})"),
    ("saved-order random E*", True, f"with $E^*$ between {lo_r} and {hi_r}."),
    ("UMAP tissue E0", True, f"tissue $E_0$ {rng(umap_t[['E0', 'E0_exact']].to_numpy().ravel(), 3)} for UMAP"),
    ("other tissue E0", True, f"against {rng(others_t[['E0', 'E0_exact']].to_numpy().ravel(), 2)} for the others"),
    ("native coherence range", True, f"(range {rng(native['C_unclipped'], 3)})"),
    ("null z range", True, f"lay {rng(perm['z'], 1)} null standard deviations"),
    ("coherence range across objects", True, f"with range across objects {rng(c_ranges, 3)}"),
    ("tie fractions", True, f"equidistant for {rng(fractions, 1)}\\% of the spots"),
    ("refit SD bound (results)", True, f"SDs of at most {bound(max(e_star_sd), 4)},"),
    ("refit SD bound (discussion)", True, f"(standard deviation at most {bound(max(e_star_sd), 4)})"),
    ("bulb interval", True, f"their means and PCA lay within {rng([bulb['mean_scvi'], bulb['mean_umap'], bulb['pca']], 3)}."),
    ("tied probe edge bound", True, f"edge survival by up to {bound(d_e, 4)}"),
    ("tied probe separation bound", True, f"separation by up to {bound(d_s, 4)} in absolute value"),
]
tail = json.loads((ROOT / "evidence/raag-v3/full-20261009-c/report.json").read_text())
smooth = {row["epsilon_decimal"]: row["actual"] for row in tail["rows"] if row["control"] == "smooth"}
first, last = smooth["1"], smooth["1E-10"]
checks += [
    ("tail coherence", round(first["allocation_coherence"], 3) == round(last["allocation_coherence"], 3) == 0.964, "coherence 0.964 at both tail masses"),
    ("tail divergence", f"{first['allocation_pair_js']:.2e}" == "2.65e-02" and f"{last['allocation_pair_js']:.2e}" == "2.65e-12", "$2.65\\times10^{-2}$ to $2.65\\times10^{-12}$"),
    ("tail total variation", f"{first['allocation_mean_pair_tv']:.3f}" == "0.158" and f"{last['allocation_mean_pair_tv']:.2e}" == "1.58e-11", "from 0.158 to $1.58\\times10^{-11}$"),
]
for label, ok, snippet in checks:
    if not ok:
        fail(f"number check failed: {label}")
    if snippet not in tex:
        fail(f"text does not contain {snippet!r} ({label})")
report["number_checks"] = len(checks)

report["errors"] = errors
report["status"] = "PASS" if not errors else "FAIL"
print(json.dumps(report, indent=1))
if "--write" in sys.argv:
    out = ROOT / "evidence/submission-20261011/audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_result.json").write_text(json.dumps(report, indent=1) + "\n")
sys.exit(1 if errors else 0)
