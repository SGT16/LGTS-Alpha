from pathlib import Path
from collections import defaultdict
from decimal import Decimal, getcontext
import csv, hashlib, json
getcontext().prec=50
root=Path(__file__).resolve().parent
p=json.loads((root/'provenance.json').read_text())
for f in p['original_files']:
    path=root/f['archive_member']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==f['sha256'],str(path)
    with path.open(newline='') as stream:
        assert len(list(csv.DictReader(stream)))==f['rows'],str(path)
groups=defaultdict(dict)
with (root/'recomputed_component_contrast/learned_squareplus_source_rows.csv').open(newline='') as stream:
    rows=list(csv.DictReader(stream))
assert len(rows)==90
for r in rows:
    key=(r['dataset'],r['model'])
    groups[key].setdefault(r['activation'],{})[r['seed']]=Decimal(r['test_f1_macro'])
assert len(groups)==9
contrasts=[]
for key,arms in sorted(groups.items()):
    a=arms['LGTS-f4-Alpha']; s=arms['LGTS-f4-Base-Squareplus-Learned']
    assert len(a)==len(s)==5 and set(a)==set(s)
    contrasts.append(sum(a[k]-s[k] for k in a)/Decimal(5))
mean=sum(contrasts)/Decimal(9)
assert sum(x>0 for x in contrasts)==4
assert mean.quantize(Decimal('0.00001'))==Decimal('-0.00980')
print(json.dumps({'status':'passed','archived_files':len(p['original_files']),'source_scores':len(rows),'paired_seeds':45,'contexts':len(groups),'alpha_higher_contexts':4,'equal_context_mean_difference':str(mean)},indent=2))
