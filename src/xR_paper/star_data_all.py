# -*- coding: utf-8 -*-
"""star_data_all.py — mesmo builder do star_data, mas para TODAS as competicoes com 360 (~91k).
Reusa star_data.process_match. Cache cache/_star_all.pkl. Para testar a confiabilidade da
metrica de jogador (xBLV) com muito mais dados. Nao altera nada existente."""
import os, sys, glob
from pathlib import Path
_root = Path(__file__).resolve().parent.parent.parent
os.chdir(str(_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'xR_experiments'))
import pandas as pd
import star_data as sd
from express_data import load_json

DATA_DIR = Path('StatsBomb_2/data')
PKL_ALL = 'cache/_star_all.pkl'


def build_all(force=False):
    if not force and os.path.exists(PKL_ALL):
        return pd.read_pickle(PKL_ALL)
    mfiles = glob.glob(str(DATA_DIR / 'matches' / '*' / '*.json'))
    mids = []
    for mf in mfiles:
        data = load_json(mf)
        if data:
            for mm in data:
                if (DATA_DIR / 'three-sixty' / (str(mm['match_id']) + '.json')).exists():
                    mids.append(mm['match_id'])
    mids = sorted(set(mids))
    print('jogos com 360 (todas competicoes):', len(mids))
    recs = []
    for j, mid in enumerate(mids):
        recs.extend(sd.process_match(mid))
        if (j + 1) % 40 == 0:
            print('  ...%d/%d jogos | %d pressoes' % (j + 1, len(mids), len(recs)))
    df = pd.DataFrame(recs)
    df.to_pickle(PKL_ALL)
    print('cache salvo:', PKL_ALL, '| pressoes', len(df))
    return df


if __name__ == '__main__':
    df = build_all(force=True)
    print('pressoes:', len(df), '| com portador id: %.3f | jogadores distintos: %d'
          % (df['carrier_id'].notna().mean(), df['carrier_id'].nunique()))
