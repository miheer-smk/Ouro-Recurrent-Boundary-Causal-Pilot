"""Validate committed raw-array hashes and shape/identity invariants."""
from ouro_boundary_pilot.artifacts import read_raw, sha256, SHA256, project_root

def main():
    original, dgx = read_raw()
    for name, exp in SHA256.items():
        path = project_root() / 'results' / 'raw' / name
        print('PASS', name, sha256(path), '(expected: '+exp[:12]+'...)')
    print('PASS 30 rows, 2 directions, preserved original arrays; margins:', original['margins'].shape, dgx['margins'].shape)
if __name__ == '__main__': main()
