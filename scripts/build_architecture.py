#!/usr/bin/env python3
"""Create the editable vector workflow schematic (no experimental data)."""
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="610" viewBox="0 0 1000 610">',
         '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" fill="#456477"/></marker></defs>',
         '<rect width="1000" height="610" fill="white"/>',
         '<g font-family="Arial" fill="#000000">']


def text(x, y, label, size=20, bold=False, anchor='start'):
    if bold and (len(label) != 1 or label not in 'abcdefghijklmnopqrstuvwxyz'):
        raise ValueError('Only standalone panel letters may use bold type')
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


text(12,27,'a',23,True)
text(43,27,'Coordinate geometry: compare matched cells',23)
for x in (42,370,698):
    box(x,43,288,163,'#f1f6fa')
text(56,69,'Matched row identities',20)
matrix(59,87,['#d1dfeb','#8bb1cf','#407ba7'])
matrix(142,87,['#dbe9ed','#85b4bd','#387b86'])
text(60,158,'Reference X',18)
text(179,158,'Candidate Z',18)
text(56,187,'Fixed arrays; no retraining',18)
line(333,125,365,125,True)
text(384,69,'Nearest-neighbour graphs',20)
graph(393,88,'#285b8a')
graph(535,88,'#477b4b',True)
text(398,158,'G(X)',19)
text(538,158,'G(Z)',19)
text(384,187,'Unweighted shortest paths',18)
line(661,125,693,125,True)
text(712,69,'Report separate diagnostics',20)
text(712,105,'R   reachability',19)
text(712,143,'E   edge survival',19)
text(712,181,'S   separation survival',19)

text(12,239,'b',23,True)
text(43,239,'Native allocations: check meaning before smoothness',23)
for x in (42,370,698):
    box(x,255,288,150,'#f5f2f8')
text(56,281,'Native probability matrix P',20)
matrix(59,302,['#e2d8eb','#b59dc8','#806095'])
text(130,315,'Rows sum to one',18)
text(130,341,'Known allocation axes',18)
text(56,382,'Continuous Z is ineligible',18)
line(333,329,365,329,True)
text(384,281,'Exact Jensen–Shannon ratio',20)
text(384,317,'D(edge) / D(all distinct pairs)',18)
text(384,346,'C = clip[1 − ratio, 0, 1]',19)
text(384,382,'Fixed reference edges',18)
line(661,329,693,329,True)
text(712,281,'Apply dispersion guards',20)
text(712,317,'Degenerate denominator?',18)
text(712,346,'Near-uniform allocation?',18)
text(712,382,'Eligible C or guarded status',18)

text(12,437,'c',23,True)
text(43,437,'Reference stress tests',22)
text(510,437,'d',23,True)
text(542,437,'Null and invariance controls',22)

# Four unboxed vector glyphs replace the text-only strip. They describe the
# protocols, not additional measurements: the null histogram is illustrative.
# Graph-scale grid: three neighbour counts crossed with three PC dimensions.
for row in range(3):
    for column in range(3):
        x, y = 111 + 23 * column, 458 + 23 * row
        parts.append(f'<rect x="{x}" y="{y}" width="19" height="19" rx="2" '
                     f'fill="{["#d1dfeb", "#8bb1cf", "#407ba7"][(row + column) % 3]}"/>')
line(103,456,103,526)
line(109,531,181,531)
text(97,497,'d',16,anchor='end')
text(145,548,'k',16,anchor='middle')
text(145,575,'Graph scale',19,anchor='middle')
text(145,598,'9 settings',16,anchor='middle')

# Reassignment: a coloured row multiset is preserved while identity changes.
cell_colours = ['#285b8a', '#c6792b', '#806095']
for index, colour in enumerate(cell_colours):
    left_y = 467 + index * 24
    right_y = 467 + ((index + 1) % 3) * 24
    parts.append(f'<circle cx="354" cy="{left_y}" r="7" fill="{colour}"/>')
    parts.append(f'<circle cx="441" cy="{right_y}" r="7" fill="{colour}"/>')
    line(365,left_y,430,right_y,True)
text(398,575,'Row reassignment',19,anchor='middle')
text(398,598,'5 fractions',16,anchor='middle')

# Permutation-null glyph: no axes or counts imply a measured distribution.
for index, height in enumerate([12, 27, 49, 64, 53, 31, 14]):
    parts.append(f'<rect x="{600 + index * 12}" y="{535 - height}" width="9" '
                 f'height="{height}" fill="#8bb1cf"/>')
line(596,537,703,537)
parts.append('<path d="M695 458L695 537" stroke="#c6792b" stroke-width="3"/>')
text(650,575,'Permutation null',19,anchor='middle')
text(650,598,'999 draws',16,anchor='middle')

# Invariance: an original triangle and its rotated/rescaled counterpart.
for points in [[(827,519),(858,472),(881,519)], [(928,469),(960,515),(911,522)]]:
    parts.append('<path d="M' + 'L'.join(f'{x} {y}' for x,y in points) +
                 'Z" fill="none" stroke="#477b4b" stroke-width="2"/>')
    for x,y in points:
        parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#477b4b"/>')
parts.append('<path d="M874 464Q897 440 922 460" fill="none" '
             'stroke="#456477" stroke-width="2" marker-end="url(#arrow)"/>')
text(900,575,'Invariance',19,anchor='middle')
text(900,598,'Rotate / scale / reorder',15,anchor='middle')
parts.append('</g></svg>')
(ROOT/'figures/architecture.svg').write_text('\n'.join(parts)+'\n')
print('Wrote editable Figure 1 workflow SVG')
