"""Fast data/ID/frozen-definition tests; no model training or inference."""
import ast
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from release_paths import check_inputs

class ReleaseTests(unittest.TestCase):
    def test_frozen_input_hashes(self):
        self.assertGreaterEqual(check_inputs(),105)

    def test_canonical_universes(self):
        human=pd.read_csv(ROOT/'data/human/oof.csv')
        external=pd.read_csv(ROOT/'data/external/matched.csv')
        mouse=pd.read_csv(ROOT/'data/mouse/predictions.csv')
        self.assertEqual((len(human),human.ENSID.nunique()),(12981,12981))
        self.assertEqual((len(external),external.ENSID.nunique()),(11381,11381))
        self.assertEqual((len(mouse),mouse.mouse_ENSID.nunique()),(13744,13744))
        self.assertEqual(mouse.ortholog_status.value_counts().to_dict(),{'ORTHOLOG_REPRESENTED':10200,'NO_CURRENT_1TO1_ORTHOLOG':3544})
        self.assertTrue(set(external.ENSID)<set(human.ENSID))
        for fold in range(5):
            groups={part:(ROOT/f'training/config/splits/fold_{fold}_{part}_ids.txt').read_text().splitlines() for part in ('train','val','test')}
            self.assertEqual(set.union(*(set(v) for v in groups.values())),set(human.ENSID))
            self.assertEqual(sum(map(len,groups.values())),12981)
            self.assertEqual(set(groups['test']),set(human.loc[human.fold==fold,'ENSID']))

    def test_complete_distillation(self):
        d=pd.read_csv(ROOT/'data/distillation/predictions.csv')
        self.assertEqual(len(d),116829)
        self.assertFalse(d.duplicated(['ENSID','seed','arm']).any())
        self.assertEqual(set(d.seed),{42,2026,3407})
        self.assertEqual(d.groupby(['seed','arm']).size().tolist(),[12981]*9)
        self.assertEqual(len(d.groupby(['seed','arm','fold'])),45)

    def test_measurement_order_and_negative_controls(self):
        m=pd.read_csv(ROOT/'data/mouse/measurement_manifest.csv')
        self.assertEqual(len(m),27)
        self.assertEqual(m.stable_experiment_index.tolist(),list(range(27)))
        self.assertEqual(m.bootstrap_seed.tolist(),list(range(42,69)))
        self.assertEqual(m.loc[m.source_dataset=='Koszinowski','n_model_matched'].tolist(),[376])
        d=pd.read_csv(ROOT/'results/mouse/M5_PER_EXPERIMENT_MODEL_DELTAS.csv')
        simon=d.loc[(d.source_dataset=='Simon')&(d.cell_type=='MEF')&(d.contrast=='FUSION_MINUS_LM')]
        self.assertEqual(len(simon),2)
        self.assertTrue((simon.delta_spearman_point<0).all())

    def test_implementation_syntax_and_local_import_closure(self):
        # Optional scientific libraries are not imported or called by this test.
        for base in ['scripts','training','tests']:
            for path in (ROOT/base).rglob('*.py'):
                source=path.read_text();compile(source,str(path),'exec')
                for node in ast.walk(ast.parse(source)):
                    if isinstance(node,ast.ImportFrom) and node.module:
                        if node.module.startswith('training.'):
                            dest=ROOT/Path(*node.module.split('.'))
                            self.assertTrue(dest.with_suffix('.py').is_file() or dest.is_dir(),f'{path.name}: {node.module}')
                        if node.module.startswith(('pd1_','pd2_','pd3_','pd4_','stage_m')):
                            self.assertTrue((path.parent/(node.module+'.py')).is_file(),f'Missing local dependency: {path.name}: {node.module}')
        cfg=json.loads((ROOT/'training/config/lm.json').read_text())['config']
        self.assertEqual(cfg['SPECIES'],['human'])
        self.assertEqual((cfg['LORA_R'],cfg['LORA_DROPOUT']),(16,0.0))
        for p in [ROOT/'environments/r/environment.yml',ROOT/'CITATION.cff']:
            self.assertIsInstance(yaml.safe_load(p.read_text()),dict)

if __name__=='__main__':unittest.main(verbosity=2)
