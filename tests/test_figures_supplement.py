"""Fast source, workbook and export checks; no new performance calculations."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import unittest
from zipfile import ZipFile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import figures_external as renderer


class SupplementTests(unittest.TestCase):
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
