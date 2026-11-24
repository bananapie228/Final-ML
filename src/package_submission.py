import argparse
import hashlib
import json
import re
import sys
import zipfile

from src.config import ROOT
from src.data import verify_processed

MANDATORY = [
    'README.md', 'REPORT.md', 'requirements.txt', 'requirements-lock.txt', 'pytest.ini',
    'notebooks/Final.ipynb', 'app/app.py', 'data/README.md', 'data/processed/manifest.json',
    'data/processed/vacancies_sample.csv.gz', 'data/processed/train_ids.csv', 'data/processed/test_ids.csv',
    'models/registry.json', 'results/tables/final_comparison.csv', 'results/summary.json',
    'presentation/Final_slides.pdf',
]
FOLDERS = ['src', 'tests', 'app', 'notebooks', 'data', 'models', 'results', 'presentation']
SKIP_PARTS = {'__pycache__', '.pytest_cache', '.ipynb_checkpoints', '.venv'}
SKIP_NAMES = {'.DS_Store'}
PLACEHOLDER = re.compile(r'TODO|\[(?:Diyar|Galymzhan)’s parts|\[What we reviewed')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(skip=()):
    files = [ROOT / p for p in MANDATORY if '/' not in p] + [ROOT / '.gitignore']
    for folder in FOLDERS:
        for path in (ROOT / folder).rglob('*'):
            if not path.is_file() or path.name in SKIP_NAMES or SKIP_PARTS & set(path.parts):
                continue
            if str(path.relative_to(ROOT)) in skip:
                continue
            if path.parent == ROOT / 'data/raw' and path.name != '.gitkeep':
                continue  # the 558 MB source file is not redistributed; data/README.md says how to get it
            files.append(path)
    return sorted(set(files))


def placeholders():
    found = []
    for rel in ['REPORT.md', 'README.md', 'presentation/speaker_notes.md']:
        if not (ROOT / rel).is_file():
            continue
        for number, line in enumerate((ROOT / rel).read_text().splitlines(), 1):
            if PLACEHOLDER.search(line):
                found.append(f'{rel}:{number}: {line.strip()[:90]}')
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build the Final submission archive.')
    parser.add_argument('--surname', default='Turemuratov')
    parser.add_argument('--name', default='Galymzhan')
    parser.add_argument('--group', default='SE2418')
    parser.add_argument('--draft', action='store_true', help='Allow TODO placeholders (not for upload)')
    parser.add_argument('--no-manifest', action='store_true',
                        help='Verify hashes but do not put SUBMISSION_MANIFEST.json in the archive')
    parser.add_argument('--no-notes', action='store_true', help='Leave presentation/speaker_notes.md out')
    args = parser.parse_args(argv)
    for value in (args.surname, args.name, args.group):
        if not re.fullmatch(r'[A-Za-z0-9-]+', value):
            raise SystemExit(f'Use only Latin letters, digits and hyphens: {value!r}')

    verify_processed()
    missing = [p for p in MANDATORY if not (ROOT / p).is_file()]
    if missing:
        raise SystemExit(f'Missing files: {missing}')
    todo = placeholders()
    if todo and not args.draft:
        raise SystemExit('Placeholders still to fill in (or pass --draft):\n' + '\n'.join(todo))

    stem = f'Final_{args.surname}_{args.name}_{args.group}' + ('_DRAFT' if todo else '')
    files = collect(skip={'presentation/speaker_notes.md'} if args.no_notes else set())
    manifest = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    manifest_path = ROOT / 'SUBMISSION_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    archive = ROOT / f'{stem}.zip'
    members = files if args.no_manifest else files + [manifest_path]
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in members:
            zf.write(path, f'{stem}/{path.relative_to(ROOT)}')
    with zipfile.ZipFile(archive) as zf:
        if zf.testzip() is not None:
            raise SystemExit('Corrupted member in archive')
        for rel, digest in manifest.items():
            if hashlib.sha256(zf.read(f'{stem}/{rel}')).hexdigest() != digest:
                raise SystemExit(f'Hash mismatch for {rel}')
    print(f'{archive.name}: {len(members)} files, {archive.stat().st_size / 1e6:.1f} MB, all hashes verified')
    if todo:
        print('DRAFT: placeholders remain:\n' + '\n'.join(todo))
    return archive


if __name__ == '__main__':
    main(sys.argv[1:])
