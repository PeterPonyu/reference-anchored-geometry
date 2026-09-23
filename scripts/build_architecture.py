#!/usr/bin/env python3
"""Create the editable vector workflow schematic (no experimental data)."""
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="490" viewBox="0 0 1000 490">',
         '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" fill="#456477"/></marker></defs>',
         '<rect width="1000" height="490" fill="white"/>',
         '<g font-family="Arial" fill="#000000">']


def text(x, y, label, size=20, bold=False, anchor='start'):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{"bold" if bold else "normal"}" text-anchor="{anchor}">{escape(label)}</text>')


def box(x, y, w, h, fill):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}" stroke="#8b9aa6" stroke-width="1.2"/>')


def line(x1, y1, x2, y2, arrow=False, dash=False):
    parts.append(f'<path d="M{x1} {y1}L{x2} {y2}" fill="none" stroke="#456477" stroke-width="2"'+
                 (' marker-end="url(#arrow)"' if arrow else '')+
                 (' stroke-dasharray="5 4"' if dash else '')+'/>')


def matrix(x, y, colours, rows=5, columns=5, size=9):
    for r in range(rows):
        for c in range(columns):
            colour = colours[(r*3+c*2) % len(colours)]
            parts.append(f'<rect x="{x+c*(size+2)}" y="{y+r*(size+2)}" width="{size}" height="{size}" fill="{colour}"/>')


def graph(x, y, colour, broken=False):
    nodes = [(0,24),(27,6),(28,46),(59,24),(90,5),(91,45)]
    for a,b in [(0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(3,5),(4,5)]:
        if broken and (a,b) in [(3,4),(3,5)]:
            continue
        ax,ay=nodes[a]; bx,by=nodes[b]
        parts.append(f'<path d="M{x+ax} {y+ay}L{x+bx} {y+by}" stroke="{colour}" stroke-width="2" fill="none"/>')
    for nx,ny in nodes:
        parts.append(f'<circle cx="{x+nx}" cy="{y+ny}" r="5" fill="{colour}"/>')


text(12,27,'A',23,True)
text(43,27,'Coordinate geometry: compare matched cells',23,True)
for x in (42,370,698):
    box(x,43,288,163,'#f1f6fa')
text(56,69,'Matched row identities',20,True)
matrix(59,87,['#d1dfeb','#8bb1cf','#407ba7'])
matrix(142,87,['#dbe9ed','#85b4bd','#387b86'])
text(60,158,'Reference X',18)
text(179,158,'Candidate Z',18)
text(56,187,'Fixed arrays; no retraining',18)
line(333,125,365,125,True)
text(384,69,'Union kNN graphs',20,True)
graph(393,88,'#285b8a')
graph(535,88,'#477b4b',True)
text(398,158,'G(X)',19)
text(538,158,'G(Z)',19)
text(384,187,'Unweighted shortest paths',18)
line(661,125,693,125,True)
text(712,69,'Report separate diagnostics',20,True)
text(712,105,'R   connected-pair survival',19)
text(712,143,'E   reciprocal edge distance',19)
text(712,181,'S   far-pair separation',19)

text(12,239,'B',23,True)
text(43,239,'Native allocations: check meaning before smoothness',23,True)
for x in (42,370,698):
    box(x,255,288,150,'#f5f2f8')
text(56,281,'Native probability matrix P',20,True)
matrix(59,302,['#e2d8eb','#b59dc8','#806095'])
text(130,315,'Rows sum to one',18)
text(130,341,'Known allocation axes',18)
text(56,382,'Continuous Z is ineligible',18)
line(333,329,365,329,True)
text(384,281,'Exact JSD normalization',20,True)
text(384,317,'D(edge) / D(all distinct pairs)',18)
text(384,346,'C = clip[1 − ratio, 0, 1]',19)
text(384,382,'Fixed reference edges',18)
line(661,329,693,329,True)
text(712,281,'Apply dispersion guards',20,True)
text(712,317,'Degenerate denominator?',18)
text(712,346,'Near-uniform allocation?',18)
text(712,382,'Eligible C or guarded status',18)

text(12,437,'C',23,True)
text(43,437,'Stress tests on the same fixed objects',22,True)
text(43,472,'9 graph settings',19)
text(272,472,'5 reassignment fractions',19)
text(581,472,'999 null draws',19)
text(793,472,'Invariance checks',19)
parts.append('</g></svg>')
(ROOT/'figures/architecture.svg').write_text('\n'.join(parts)+'\n')
print('Wrote editable Figure 1 workflow SVG')
