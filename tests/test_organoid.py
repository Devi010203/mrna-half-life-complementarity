"""Fast checks for organoid source correspondence, immutable references and portability."""
from pathlib import Path
import importlib.util
import json
import sys
import unittest
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/organoid'))
from inputs import verify_inputs,output_path
spec=importlib.util.spec_from_file_location('organoid_reproduce',ROOT/'scripts/organoid/reproduce.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class OrganoidTests(unittest.TestCase):
    def test_frozen_sources_and_protocols(self):
        self.assertGreaterEqual(verify_inputs(),40)
        record=module.source_check()
        self.assertEqual(record['control_qc_n'],5369)
    def test_registered_sets_and_directions(self):
        root=ROOT/'data/organoid/frozen/preflight_20261102/audit'
        primary=pd.read_csv(root/'primary_gene_manifest.csv')
        common=pd.read_csv(root/'three_target_gene_manifest.csv')
        self.assertEqual((len(primary),len(common)),(4890,4801))
        self.assertEqual(primary.ENSID.tolist(),sorted(primary.ENSID))
        self.assertTrue(set(common.ENSID)<set(primary.ENSID))
        folds=pd.read_csv(ROOT/'results/organoid/Fold_diagnostics.csv')
        self.assertEqual(folds.fold.tolist(),list(range(5)))
        self.assertTrue(folds.Fusion_minus_XGB.gt(0).all())
        self.assertEqual(int(folds.n.sum()),4890)
    def test_protected_output(self):
        for p in ['data/organoid','results/organoid','outputs','../outside']:
            with self.assertRaises((ValueError,FileExistsError)):
                output_path(p)
    def test_supplement_original_source_unchanged(self):
        original=json.loads((ROOT/'data/workbook_spec.json').read_text())
        self.assertEqual(len(original['workbooks'][0]['sheets']),24)
        extension=json.loads((ROOT/'data/organoid/workbook_extension.json').read_text())
        self.assertEqual([t['name'] for t in extension['tables']],['S13a','S13b','S14a','S14b','S14c','S14d','S14e','S14f','S14g'])

if __name__=='__main__':unittest.main(verbosity=2)
