"""Recompute in a disposable directory; compare outputs without overwriting the experiment."""
from pathlib import Path
import os,shutil,subprocess,sys,tempfile
import pandas as pd
root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='isc-draw-reproduce-') as d:
    dest=Path(d)
    for name in ['analysis','experiments']:
        shutil.copytree(root/name,dest/name,ignore=shutil.ignore_patterns('__pycache__'))
    (dest/'data/raw').mkdir(parents=True);(dest/'reports').mkdir();(dest/'artifacts').mkdir()
    shutil.copy2(root/'data/raw/cached_fixtures.csv',dest/'data/raw/cached_fixtures.csv')
    env=os.environ.copy();env['MPLCONFIGDIR']=str(root/'.cache/matplotlib')
    for stage in ['develop','test']:
        subprocess.run([sys.executable,'-m','experiments.draw_refinement.run','--stage',stage],cwd=dest,env=env,check=True)
    for f in ['nested_summary.csv','nested_predictions.csv','outer_splits.csv','test_metrics.csv','test_predictions.csv','final_vif.csv']:
        old=pd.read_csv(root/'reports/draw_refinement'/f);new=pd.read_csv(dest/'reports/draw_refinement'/f)
        pd.testing.assert_frame_equal(old,new,check_exact=False,rtol=1e-8,atol=1e-10)
    print('Reproduction matches all six audited output tables. Original experiment unchanged.')
