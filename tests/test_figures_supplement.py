"""Fast source, workbook and export checks; no new performance calculations."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tomllib
import unittest
from zipfile import ZipFile
from xml.etree import ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import figures_external as renderer


class SupplementTests(unittest.TestCase):
    def test_supplement_author_order(self):
        expected=['Xu Jin','Bihang Cui','Ruxin Yan','Yingchen Mao','Dinglin Zhang']
        self.assertEqual(json.loads((ROOT/'docs/publication.json').read_text())['authors'],expected)
        package=tomllib.loads((ROOT/'pyproject.toml').read_text())
        self.assertEqual([a['name'] for a in package['project']['authors']],expected)
        with ZipFile(ROOT/'supplementary/source/ESM_1_Supplementary_Information.docx') as z:
            document=ET.fromstring(z.read('word/document.xml'))
            ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            paragraphs=[''.join(t.text or '' for t in p.findall('.//w:t',ns)) for p in document.findall('.//w:p',ns)]
            self.assertEqual([p for p in paragraphs if 'Xu Jin' in p],[', '.join(expected)])
            core=ET.fromstring(z.read('docProps/core.xml'))
            self.assertEqual(core.find('{http://purl.org/dc/elements/1.1/}creator').text,'; '.join(expected))
        self.assertIn('Authors: '+', '.join(expected)+'.',(ROOT/'supplementary/source/SUPPLEMENTARY_EN.md').read_text())
        from tables import verify
        self.assertEqual(verify(ROOT,ROOT/'supplementary')['cells_checked'],5311)

    def test_saved_sources_and_readable_tables(self):
        data = renderer.load(ROOT)
        renderer.check_workbooks(ROOT, data)
        self.assertEqual(len(data['external']), 11381)
        self.assertEqual({r['n'] for r in data['models']}, {'4801'})
        self.assertEqual({r['n'] for r in data['deltas']}, {'4801'})

    def test_print_exports(self):
        for figure in ['Figure_2', 'Figure_S1', 'Figure_S2']:
            for ext in ['png', 'pdf', 'svg']:
                self.assertGreater((ROOT/'figures'/f'{figure}.{ext}').stat().st_size, 1000)
            with Image.open(ROOT/'figures'/f'{figure}.png') as im:
                self.assertTrue(all(abs(x-600)<.01 for x in im.info['dpi']))
        with Image.open(ROOT/'figures/Figure_2.png') as im:
            self.assertEqual(im.size, (4322,5480))

    def test_supplement_image_and_cross_references(self):
        with ZipFile(ROOT/'supplementary/source/ESM_1_Supplementary_Information.docx') as z:
            xml=z.read('word/document.xml').decode()
            self.assertEqual(xml.count('<w:drawing>'), 2)
            self.assertEqual(xml.count('<w:tbl>'), 33)
            self.assertIn('Figure 2b,c; Table S14a–c', xml)
            self.assertIn('Fig. S2 Gene-level predictions against External-PC1', xml)
            self.assertEqual(z.read('word/media/Figure_S2.png'), (ROOT/'figures/Figure_S2.png').read_bytes())
        text=(ROOT/'supplementary/source/SUPPLEMENTARY_EN.md').read_text()
        self.assertIn('### Fig. S2', text)
        self.assertIn('../../figures/Figure_S2.png', text)
        self.assertIn('S13b. Primary and secondary paired model differences', text)
        for required in ['References cited in this supplementary material correspond to those in the main manuscript.',
                         'separately trained Bio-XGB interpretation models',
                         'DOI: 10.5281/zenodo.16884513', 'Bolstad 2017']:
            self.assertIn(required,text)
        for obsolete in ['Reference numbers refer to', 'Bolstad 2025', '(Vock 2025)', 'For PD4', 'analyses (PD4)']:
            self.assertNotIn(obsolete,text)

    def test_file_manifest(self):
        with (ROOT/'docs/file_manifest.csv').open(newline='') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),len({r['path'] for r in rows}))
        for row in rows:
            path=ROOT/row['path']
            self.assertTrue(path.is_file(), row['path'])
            self.assertEqual(path.stat().st_size, int(row['bytes']), row['path'])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row['sha256'], row['path'])
            self.assertTrue(row['license_scope'], row['path'])
        recorded={r['path'] for r in rows}
        self.assertTrue({'figures/Figure_S2.png','scripts/figures_external.py'}<=recorded)


if __name__=='__main__':
    unittest.main(verbosity=2)
