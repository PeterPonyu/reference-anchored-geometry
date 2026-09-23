from pathlib import Path
import csv
import hashlib
import json
import numpy as np

paper = Path(__file__).resolve().parents[1]
source = paper / 'evidence/measurement-inventory/calibration_arrays/seed42.npz'
output = paper / 'scripts/plot-inputs'
output.mkdir(exist_ok=True)
candidates = ['faithful', 'shatter_0.5', 'shatter_2.5', 'collapse_0', 'noise_0.18', 'permuted']
with np.load(source, allow_pickle=False) as arrays:
    with (output / 'coordinates.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['candidate', 'cell', 'branch', 'coordinate_1', 'coordinate_2'])
        for candidate in candidates:
            for index, coordinates in enumerate(arrays[candidate]):
                writer.writerow([candidate, index, int(arrays['branch'][index]), *coordinates])
manifest = {
    'source': str(source.relative_to(paper)),
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'output_sha256': hashlib.sha256((output / 'coordinates.csv').read_bytes()).hexdigest(),
    'rows': 2700,
    'transformation': 'Lossless CSV export of six retained seed-42 coordinate arrays and original branch identities.'
}
(output / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Exported 2700 coordinate rows; no measurements recomputed.')
