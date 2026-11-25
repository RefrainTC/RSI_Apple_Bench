# Same check as the Dockerfile's post-install RUN block: every lock pin installed at the pinned version, no sglang dist.
import re, sys
from importlib import metadata
pins = {}
for raw in open(sys.argv[1]):
    raw = raw.split(' #')[0].strip()
    m = re.match(r'^([A-Za-z0-9._-]+)\s*={2,3}\s*(\S+)$', raw)
    if m:
        pins[m.group(1).lower().replace('_', '-')] = m.group(2)
inst = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in metadata.distributions() if d.metadata['Name']}
missing = sorted(n for n in pins if n not in inst)
wrong = sorted(f'{n}: want {v}, have {inst[n]}' for n, v in pins.items() if n in inst and inst[n] != v)
print(f'lock pins {len(pins)}, installed {len(inst)}, missing {missing}, wrong {wrong}')
print('sglang installed as dist:', 'sglang' in inst)
