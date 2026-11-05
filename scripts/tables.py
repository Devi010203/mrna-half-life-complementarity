#!/usr/bin/env python3
"""Repackage stored tables; never calculate statistical or model results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


STATISTICS = [
    ('Human_Pooled', 'human_pooled_oof_metrics.csv', 'S9a: human pooled OOF metrics'),
    ('Human_Paired_Deltas', 'human_pooled_oof_paired_deltas.csv', 'S9b: paired human pooled differences'),
    ('Human_Five_Fold', 'human_five_fold_summary.csv', 'S9c: unweighted five-fold mean and population SD'),
    ('Human_All_Folds', 'human_fold_metrics.csv', 'S9c: individual fold metrics'),
    ('External_Models', 'external_human_metrics.csv', 'S10a: five-model external-human evaluation'),
    ('External_Paired_Deltas', 'external_human_paired_deltas.csv', 'S10b: paired external-human differences'),
    ('Coverage_Groups', 'coverage_status_counts.csv', 'S11a: observed coverage groups'),
    ('Coverage_Internal_OOF', 'coverage_internal_oof_metrics.csv', 'S11b: INTERNAL original-target performance, not external performance for unmatched genes'),
    ('Coverage_Descriptions', 'coverage_descriptive_statistics.csv', 'S11c: descriptive summaries and missingness'),
    ('Coverage_Differences', 'coverage_standardized_differences.csv', 'S11c: stored standardized differences'),
]
PROVENANCE = [
    ('External_Datasets', 'external_dataset_provenance.csv', 'S12a: dataset-level evidence and qualifications'),
    ('External_Samples', 'external_sample_provenance.csv', 'S12a: all 64 candidate samples, including the two excluded samples'),
    ('Mabin_Accessions', 'mabin_accession_crosswalk.csv', 'S12b: tentative alias crosswalk; not file-checksum proof'),
    ('Citation_Verification', 'citation_verification.csv', 'S12c: original and source-verified citations'),
    ('Saluki_Human_Sources', 'saluki_human_source_inventory.csv', 'S12: published human source inventory'),
    ('Saluki_Script_Crosswalk', 'saluki_script_source_crosswalk.csv', 'S12: source-script input correspondence'),
]
NAMES = {
    'display': 'ESM_2_Supplementary_Tables_S1-S14.xlsx',
    'statistics': 'ESM_3a_Source_Statistics_S9-S11.xlsx',
    'provenance': 'ESM_3b_Source_Provenance_S12.xlsx',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.reader(f))


def append_exact(ws, values):
    ws.append(values)
    for c, value in zip(ws[ws.max_row], values):
        if isinstance(value, str):
            if len(value) > 32767:
                raise ValueError('Excel cell-length limit exceeded; refusing truncation')
            c.data_type = 's'  # No formula evaluation or accession/date coercion.
            c.number_format = '@'


def style(ws, color='24556D', wide=False):
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = 'B2' if ws.title == 'Index' else 'A2'
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.sheet_properties.tabColor = color
    ws.auto_filter.ref = ws.dimensions
    ws.print_title_rows = '1:1'
    ws.page_setup.orientation = 'landscape'
    ws.page_setup.paperSize = ws.PAPERSIZE_A4
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    ws.print_options.horizontalCentered = True
    for row in ws:
        for c in row:
            c.font = Font(name='Calibri', size=11)
            c.alignment = Alignment(vertical='top', wrap_text=True)
            if c.row == 1:
                c.fill = PatternFill('solid', fgColor=color)
                c.font = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
                c.border = Border(bottom=Side(style='thin', color=color))
            elif c.row % 2 == 0:
                c.fill = PatternFill('solid', fgColor='F1F5F7')
    ws.row_dimensions[1].height = 34
    for n, col in enumerate(ws.columns, 1):
        width = max(len(str(c.value or '')) for c in col)
        ws.column_dimensions[get_column_letter(n)].width = min(70 if wide else 48, max(14, width + 2))
    for row in ws.iter_rows(min_row=2):
        lines = max((len(str(c.value or '')) // max(1, int(ws.column_dimensions[c.column_letter].width)-2) + 1 for c in row), default=1)
        ws.row_dimensions[row[0].row].height = min(120, max(23, lines * 15))
    ws.oddFooter.center.text = 'Page &P of &N'


def base_book(title, explanation, metadata):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    ws = wb.create_sheet('README')
    notes = [
        ['Field', 'Description'], ['Workbook', title],
        ['Article', metadata['title']], ['Authors', ', '.join(metadata['authors'])],
        ['Contents', explanation],
        ['Data type', 'Reported study results and their source tables.'],
        ['Precision', 'Source-CSV workbooks store EVERY original CSV field as text, including decimal values and missing-value tokens. This prevents Excel 15-digit rounding and automatic identifier/date conversions. Do not coerce these cells to numbers for official reproduction; use preserved CSV inputs.'],
        ['Display tables', 'S1-S12 contain the manuscript display strings and numeric cells.'],
        ['Navigation', 'Index links to sheets; headers are frozen and filters are enabled. Source filenames and SHA256 hashes are on Index.'],
        ['Original files', 'Machine-readable CSV inputs are included in the repository.'],
        ['Interpretation', 'Primary and post hoc estimates, five-fold means, pooled OOF and ensemble estimates are distinct; retain the source estimand labels.'],
        ['Negative findings', 'All rows, including Simon MEF negative contrasts, Koszinowski n=376 and distillation limits, are retained without sorting by performance.'],
        ['Rights', 'Original software: MIT. Data and third-party terms: see DATA_LICENSES.md in the repository.'],
    ]
    for row in notes:
        append_exact(ws, row)
    style(ws, wide=True)
    ws.column_dimensions['A'].width = 23
    ws.column_dimensions['B'].width = 105
    for row in ws.iter_rows(min_row=2):
        ws.row_dimensions[row[0].row].height = max(28, (len(str(row[1].value)) // 100 + 1) * 16)
    index = wb.create_sheet('Index')
    append_exact(index, ['Sheet', 'Contents / manuscript relation', 'Rows excluding header', 'Columns', 'Source path (relative to repository)', 'Source SHA256'])
    wb.properties.title = title
    wb.properties.subject = metadata['title']
    wb.properties.creator = '; '.join(metadata['authors'])
    wb.properties.description = 'Repackaged frozen data. Not newly calculated results.'
    return wb


def add_index(wb, name, title, rows, cols, source, digest):
    ws = wb['Index']
    append_exact(ws, [name, title, rows, cols, source, digest])
    c = ws.cell(ws.max_row, 1)
    c.hyperlink = f"#'{name}'!A1"


def finish(wb, output):
    if output.exists():
        raise FileExistsError(f'Refusing overwrite: {output}')
    style(wb['Index'], wide=True)
    wb['Index'].column_dimensions['A'].width = 29
    wb['Index'].column_dimensions['B'].width = 68
    wb['Index'].column_dimensions['C'].width = 16
    wb['Index'].column_dimensions['D'].width = 12
    wb['Index'].column_dimensions['E'].width = 70
    wb['Index'].column_dimensions['F'].width = 68
    for row in wb['Index'].iter_rows(min_row=2):
        row[0].font = Font(name='Calibri', size=11, color='0563C1', underline='single')
    wb.save(output)


def normalized_rows(ws):
    return [[v if v is not None else '' for v in row] for row in ws.values]


def combined_spec(repo):
    # Preserve the original frozen specification; append the organoid extension.
    spec = json.loads((repo/'data/workbook_spec.json').read_text())
    extension = json.loads((repo/'data/organoid/workbook_extension.json').read_text())
    spec['workbooks'][0]['file'] = NAMES['display']
    spec['workbooks'][0]['sheets'] += [
        {'sheet': t['name'], 'source': extension['source'], 'source_sheet': t['name'],
         'kind': 'xlsx', 'sha256': extension['sha256']} for t in extension['tables']]
    return spec


def verify(repo, directory):
    if not __debug__:
        raise RuntimeError('Do not use python -O for workbook verification')
    spec = combined_spec(repo)
    report = {'status': 'PASS', 'model_or_statistical_calculation': False, 'workbooks': [], 'cells_checked': 0}
    for book in spec['workbooks']:
        # Fully load normal-mode workbooks before closing the underlying handle.
        # Explicit ownership avoids leaked handles with the frozen openpyxl version.
        with (directory/book['file']).open('rb') as stream:
            wb = openpyxl.load_workbook(stream, read_only=False, data_only=False)
        assert wb.sheetnames == ['README', 'Index'] + [s['sheet'] for s in book['sheets']]
        index_rows = {row[0]: row for row in list(wb['Index'].values)[1:]}
        cells = 0
        for s in book['sheets']:
            p = repo/s['source']
            assert sha(p) == s['sha256'], p
            index_source = s['source'] + ('#'+s['source_sheet'] if s['kind']=='xlsx' else '')
            assert index_rows[s['sheet']][4:6] == (index_source, s['sha256']), (book['file'], s['sheet'], 'source index mismatch')
            if s['kind'] == 'xlsx':
                origin = openpyxl.load_workbook(p, read_only=True, data_only=False)
                want = normalized_rows(origin[s['source_sheet']])
                origin.close()
            else:
                want = read_csv(p)
            got = normalized_rows(wb[s['sheet']])
            assert got == want, (book['file'], s['sheet'], 'cell mismatch')
            assert not any(c.data_type == 'f' for row in wb[s['sheet']] for c in row)
            cells += sum(len(row) for row in want)
        report['cells_checked'] += cells
        report['workbooks'].append({'file': book['file'], 'data_sheets': len(book['sheets']), 'cells': cells, 'sha256': sha(directory/book['file'])})
        wb.close()
    return report


def build(repo, output):
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((repo/'docs/publication.json').read_text())
    spec = {'version': 1, 'no_new_scientific_calculation': True, 'workbooks': []}
    original = repo/'data/sources/supplementary_display.xlsx'
    base = openpyxl.load_workbook(original, read_only=True, data_only=False)
    titles = {r[0]:r[1] for r in list(base['Index'].values)[1:] if r[0] in base.sheetnames}
    wb = base_book('Supplementary Tables S1-S14', 'Original S1-S12 preserved; S13-S14 add the frozen organoid evaluation.', metadata)
    for row in wb['README']:
        if row[0].value == 'Display tables':
            row[1].value = 'S1-S12 preserve the original cells; S13-S14 show rounded frozen organoid results. Full precision remains in results/organoid/. See S7 for methods.'
    book = {'file': NAMES['display'], 'sheets': []}
    for name in base.sheetnames:
        if name == 'Index':
            continue
        rows = list(base[name].values)
        ws = wb.create_sheet(name)
        for row in rows:
            append_exact(ws, list(row))
        style(ws, color='24556D' if int(name[1:].rstrip('abc')) < 9 else '665280')
        src = original.relative_to(repo).as_posix()
        add_index(wb, name, titles[name], len(rows)-1, len(rows[0]), src+'#'+name, sha(original))
        book['sheets'].append({'sheet': name, 'source': src, 'source_sheet': name, 'kind': 'xlsx', 'sha256': sha(original)})
    extension = json.loads((repo/'data/organoid/workbook_extension.json').read_text())
    added = openpyxl.load_workbook(repo/extension['source'], read_only=True, data_only=False)
    assert sha(repo/extension['source']) == extension['sha256']
    for t in extension['tables']:
        name = t['name']; rows = list(added[name].values)
        ws = wb.create_sheet(name)
        for row in rows:
            append_exact(ws, list(row))
        style(ws, color='665280')
        add_index(wb, name, name+': '+t['title']+'; Supplementary Methods S7', len(rows)-1,
                  len(rows[0]), extension['source']+'#'+name, extension['sha256'])
        book['sheets'].append({'sheet': name, 'source': extension['source'], 'source_sheet': name,
                              'kind': 'xlsx', 'sha256': extension['sha256']})
    added.close()
    finish(wb, output/NAMES['display'])
    spec['workbooks'].append(book)
    base.close()
    for group, definitions in [('statistics', STATISTICS), ('provenance', PROVENANCE)]:
        wb = base_book('Stored source '+group, 'Supplementary S9-S11 statistics' if group == 'statistics' else 'Supplementary S12 provenance, including qualifications and exclusions', metadata)
        book = {'file': NAMES[group], 'sheets': []}
        for name, filename, title in definitions:
            p = repo / ('results/unified' if group == 'statistics' else 'data/sources') / filename
            rows = read_csv(p)
            ws = wb.create_sheet(name)
            for row in rows:
                append_exact(ws, row)
            style(ws, color='665280' if group == 'statistics' else '496954')
            src = p.relative_to(repo).as_posix()
            add_index(wb, name, title, len(rows)-1, len(rows[0]), src, sha(p))
            book['sheets'].append({'sheet': name, 'source': src, 'kind': 'csv', 'sha256': sha(p)})
        finish(wb, output/NAMES[group])
        spec['workbooks'].append(book)
    # Specification is supplied read-only; regeneration must agree.
    spec_path = repo/'data/workbook_spec.json'
    if spec_path.exists():
        assert combined_spec(repo) == spec
    else:
        spec_path.write_text(json.dumps(spec, indent=2)+'\n')
    return verify(repo, output)


def main():
    if not __debug__:
        raise RuntimeError('Do not use python -O for workbook verification')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New directory inside outputs/; default outputs/workbooks_1')
    parser.add_argument('--check', action='store_true', help='Validate supplied workbooks; no writes')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.check:
        report = verify(repo, repo/'supplementary')
    else:
        out = (repo/(args.output or Path('outputs/workbooks_1'))).resolve()
        if not out.is_relative_to(repo/'outputs') or out==repo/'outputs' or out.exists():
            raise ValueError('Output must be a new child directory inside outputs/')
        report = build(repo, out)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
