"""Audit a downloaded official SAFE ZIP without modifying its contents."""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

p = argparse.ArgumentParser()
p.add_argument('archive', type=Path)
p.add_argument('catalog', type=Path)
p.add_argument('report', type=Path)
a = p.parse_args()
catalog = json.loads(a.catalog.read_text(encoding='utf-8-sig'))
digests = {name: hashlib.new(name) for name in ('md5', 'sha256')}
with a.archive.open('rb') as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
        for h in digests.values():
            h.update(block)
declared = next(x['Value'] for x in catalog['Checksum'] if x['Algorithm'] == 'MD5')
with zipfile.ZipFile(a.archive) as z:
    names = z.namelist()
    unsafe = [n for n in names if n.startswith(('/', '\\')) or '..' in Path(n).parts]
    if unsafe:
        raise ValueError('Unsafe archive members')
    bad_member = z.testzip()
    metadata_names = [n for n in names if n.endswith('/MTD_MSIL2A.xml')]
    if len(metadata_names) != 1:
        raise ValueError('Expected exactly one MTD_MSIL2A.xml')
    xml = ET.fromstring(z.read(metadata_names[0]))
    metadata = {}
    for e in xml.iter():
        key = e.tag.split('}')[-1]
        if key in ('BOA_QUANTIFICATION_VALUE', 'BOA_ADD_OFFSET', 'PROCESSING_BASELINE',
                   'PRODUCT_START_TIME', 'PRODUCT_URI'):
            metadata.setdefault(key, []).append({'value': e.text, 'attributes': e.attrib})
    band_members = [n for n in names if re.search(r'_B(?:0[1-9]|1[12]|8A)_\d+m\.jp2$', n)]
    scl_members = [n for n in names if re.search(r'_SCL_\d+m\.jp2$', n)]
report = {
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'archive': str(a.archive.resolve()), 'catalog': str(a.catalog.resolve()),
    'catalog_product': catalog['Name'], 'bytes': a.archive.stat().st_size,
    'declared_bytes': catalog['ContentLength'],
    'local_hashes': {k: v.hexdigest() for k, v in digests.items()},
    'declared_md5': declared, 'md5_matches': digests['md5'].hexdigest() == declared,
    'byte_count_matches': a.archive.stat().st_size == catalog['ContentLength'],
    'zip_crc_passed': bad_member is None, 'bad_member': bad_member,
    'safe_root_matches': all(n.startswith(catalog['Name'] + '/') for n in names),
    'member_count': len(names), 'spectral_members': band_members,
    'scl_members': scl_members, 'metadata': metadata,
    'model_ready': False,
    'note': 'Archive audit only; no crop, alignment, normalization or model inference.'
}
a.report.parent.mkdir(parents=True, exist_ok=True)
a.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: report[k] for k in ('bytes', 'md5_matches', 'byte_count_matches',
                                      'zip_crc_passed', 'safe_root_matches', 'metadata')},
                 ensure_ascii=False, indent=2))
if not all(report[k] for k in ('md5_matches', 'byte_count_matches', 'zip_crc_passed', 'safe_root_matches')):
    raise SystemExit('Archive verification needs review')
