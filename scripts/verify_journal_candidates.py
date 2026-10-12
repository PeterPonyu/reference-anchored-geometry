#!/usr/bin/env python3
"""Check and render local Scientific Reports document candidates.

This inspects document bytes and citation coverage. It does not fit models,
review scientific validity, certify biological validation, or submit an article.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import fitz
from PIL import Image, ImageDraw


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def supplementary_numbers(text, kind):
    noun = r'(?:Figures?|Figs?\.?)' if kind == 'figure' else r'Tables?'
    found = set()
    pattern = (r'Supplementary\s+' + noun + r'\s*~?\s*'
               r'(S\d+(?:\s*(?:--|-|to|and|,)\s*S?\d+)*)')
    for match in re.finditer(pattern, text):
        found.update(int(n) for n in re.findall(r'S?(\d+)', match[1]))
        for start, end in re.findall(r'S?(\d+)\s*(?:--|-|to)\s*S?(\d+)', match[1]):
            assert int(start) <= int(end)
            found.update(range(int(start), int(end) + 1))
    return found


def check(root):
    root = root.resolve()
    base = root / 'source' if (root / 'source/scientific-reports.json').exists() else root
    cfg = json.loads((base / 'scientific-reports.json').read_text())
    report_path = (base / cfg['qa']).resolve() / 'format-check.json'
    report = json.loads(report_path.read_text())
    assert report['status'] == 'PASS'
    assert report['generator_sha256'] == digest(base / 'scripts/build_scientific_reports.py')
    assert report['configuration_sha256'] == digest(base / 'scientific-reports.json')
    assert report['canonical_source_sha256'] == digest(base / cfg['main_source'])
    if cfg['supplement_source']:
        assert report['canonical_supplement_sha256'] == digest(base / cfg['supplement_source'])
    assert report['abstract_words'] <= 200
    assert report['main_figures'] + report['main_tables'] <= 8
    main = (base / cfg['main_source']).with_name('paper_srep.tex').read_text()
    supporting = (base / cfg['main_source']).with_name('supporting_srep.tex').read_text()
    assert main.index(r'\section*{Data Availability}') < main.index(r'\bibliography{')
    assert main.index(r'\bibliography{') < main.index(r'\section*{Author contributions}')
    assert main.index(r'\section*{Author contributions}') < main.index(r'\section*{Figures and tables}')
    assert r'\title{' + cfg['title'] + '}' in main
    assert r'\title{' + cfg['title'] + '}' in supporting
    coverage = {}
    for kind in ('figure', 'table'):
        expected = set(range(1, report['supplementary_' + kind + 's'] + 1))
        cited = supplementary_numbers(main, kind)
        coverage[kind] = {'expected': sorted(expected), 'cited': sorted(cited),
                          'missing': sorted(expected - cited), 'unknown': sorted(cited - expected)}
        assert not cited - expected, (kind, cited - expected)
        assert not expected - cited, (kind, 'uncited supplementary items', expected - cited)
    qa = root / 'qa/resumed-document-review'
    qa.mkdir(parents=True, exist_ok=True)
    documents = {}
    for name, record in report['documents'].items():
        pdf = (base / cfg['output']).resolve() / ('scientific-reports-' + name + '.pdf')
        assert digest(pdf) == record['sha256']
        out = qa / (name + '-' + record['sha256'][:12])
        out.mkdir(exist_ok=True)
        doc = fitz.open(pdf)
        assert len(doc) == record['pages']
        page_records, out_of_bounds, figure_pages, declaration_pages = [], [], [], []
        for number, page in enumerate(doc, 1):
            text = page.get_text()
            assert text.strip(), (pdf, number, 'blank page')
            assert '??' not in text and '\ufffd' not in text, (pdf, number, 'unresolved text')
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines', []):
                    for span in line['spans']:
                        x0, y0, x1, y1 = span['bbox']
                        if min(x0, y0) < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1:
                            out_of_bounds.append({'page': number, 'text': span['text'], 'bbox': span['bbox']})
            image = out / f'page-{number:03d}.png'
            page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False).save(image)
            page_records.append({'page': number, 'path': str(image.relative_to(root)), 'sha256': digest(image)})
            if re.search(r'(?:Figure|Fig\.)\s*S?\d+\s*[.:]', text):
                figure_pages.append(number)
            if 'Author contributions' in text or 'Additional Information' in text:
                declaration_pages.append(number)
        assert not out_of_bounds, (pdf, out_of_bounds)
        fonts = subprocess.run(['pdffonts', str(pdf)], capture_output=True, text=True, check=True).stdout
        assert all(row.split()[-5] == 'yes' for row in fonts.splitlines()[2:] if row.strip()), fonts
        (out / 'fonts.txt').write_text(fonts)
        contacts = []
        for start in range(0, len(page_records), 8):
            selected = page_records[start:start + 8]
            width, height = 375, 550
            sheet = Image.new('RGB', (4 * width, 2 * height), '#e6e9ed')
            draw = ImageDraw.Draw(sheet)
            for offset, record in enumerate(selected):
                image = Image.open(root / record['path']).convert('RGB')
                image.thumbnail((width - 8, height - 30))
                x, y = (offset % 4) * width, (offset // 4) * height
                draw.text((x + 8, y + 8), f'{root.name[:18]} {name} p{record["page"]}', fill='#172c40')
                sheet.paste(image, (x + (width - image.width) // 2, y + 28))
            target = out / f'overview-{start // 8 + 1:02d}.jpg'
            sheet.save(target, quality=88)
            contacts.append({'path': str(target.relative_to(root)), 'sha256': digest(target),
                             'pages': [p['page'] for p in selected]})
        documents[name] = {'pdf': str(pdf.relative_to(root)), 'sha256': digest(pdf),
                           'pages': len(doc), 'page_images': page_records, 'overview_sheets': contacts,
                           'figure_caption_pages': figure_pages, 'declaration_pages': declaration_pages,
                           'embedded_fonts': True, 'text_within_page_bounds': True}
    receipt = {'mechanical_status': 'PASS', 'review_status': 'rendered; manual review pending',
               'format_report_sha256': digest(report_path), 'verifier_sha256': digest(Path(__file__)),
               'supplementary_citation_coverage': coverage, 'documents': documents,
               'scope': 'Document hashes, references, declaration order, embedded fonts and page bounds; no model fitting or scientific peer review.'}
    (qa / 'render-manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'root': str(root), 'status': 'PASS', 'pages': sum(d['pages'] for d in documents.values()),
                      'supplementary_citation_missing': {k: v['missing'] for k, v in coverage.items()}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', type=Path, nargs='+')
    args = parser.parse_args()
    for root in args.roots:
        check(root)
