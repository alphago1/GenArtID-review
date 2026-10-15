"""Fetch exact public dependencies from their official Hub; never load weights."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value or ':' in value:
        raise ValueError('Expected a relative artifact path')
    return path


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    identifiers = set()
    destinations = set()
    for asset in manifest['assets']:
        if asset['id'] in identifiers:
            raise ValueError('Duplicate asset identifier')
        identifiers.add(asset['id'])
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', asset['repo']):
            raise ValueError('Invalid public repository identifier')
        if not re.fullmatch(r'[0-9a-f]{40}', asset['revision']):
            raise ValueError('A full immutable commit hash is required')
        directory = relative_path(asset['directory'])
        for name, row in asset['files'].items():
            dest = str(directory / relative_path(name))
            if dest in destinations or not re.fullmatch(r'[0-9a-f]{64}', row['sha256']):
                raise ValueError('Duplicate destination or invalid SHA256')
            if not isinstance(row['bytes'], int) or row['bytes'] <= 0:
                raise ValueError('Invalid frozen file size')
            destinations.add(dest)
    return manifest


def asset_directory(output, asset):
    output = Path(output).resolve()
    directory = output.joinpath(*relative_path(asset['directory']).parts).resolve()
    if not directory.is_relative_to(output):
        raise ValueError('Asset directory escapes output')
    return directory


def verify_asset(output, asset):
    directory = asset_directory(output, asset)
    for name, row in asset['files'].items():
        path = directory.joinpath(*relative_path(name).parts)
        if not path.resolve().is_relative_to(directory):
            raise ValueError('Asset file escapes its directory')
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Missing file or checksum mismatch: ' + str(path))


def download_asset(output, asset):
    # Import the downloader only for an explicit --download invocation.
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=asset['repo'], revision=asset['revision'],
                      local_dir=asset_directory(output, asset), allow_patterns=list(asset['files']),
                      endpoint='https://huggingface.co', token=False)
    verify_asset(output, asset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('model_assets.json'))
    parser.add_argument('--output', type=Path, default=Path('.'))
    parser.add_argument('--asset', choices=['glm', 'text2vec', 'all'], default='glm')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-manifest', action='store_true', help='Check metadata only, no network or models')
    mode.add_argument('--download', action='store_true', help='Download from official Hub and verify SHA256')
    mode.add_argument('--verify', action='store_true', help='Verify existing local files without network')
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    assets = [a for a in manifest['assets'] if args.asset == 'all' or a['id'] == args.asset]
    if not assets:
        raise ValueError('No assets selected')
    if not args.check_manifest:
        for asset in assets:
            (download_asset if args.download else verify_asset)(args.output, asset)
    print(json.dumps({'status': 'MANIFEST_VALID' if args.check_manifest else 'ASSETS_VERIFIED',
                      'assets': [a['id'] for a in assets], 'files': sum(len(a['files']) for a in assets),
                      'bytes': sum(row['bytes'] for a in assets for row in a['files'].values()),
                      'models_loaded': False}))


if __name__ == '__main__':
    main()
