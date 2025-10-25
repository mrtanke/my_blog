import os
import re
from urllib.parse import unquote
import zipfile
import shutil
import datetime
import yaml
from pathlib import Path


def unzip_notion() -> Path:
    """Scan `tmp_zip/` for the single .zip file, extract it there, and return the extracted folder Path.

    Expects exactly one .zip file under `tmp_zip/`. The extracted folder will be
    created as `tmp_zip/<zip-stem>/`.
    """
    tmp_dir = Path("tmp_zip")
    if not tmp_dir.exists():
        raise FileNotFoundError("Temporary folder not found: tmp_zip/")
    zips = list(tmp_dir.glob('*.zip'))
    if len(zips) == 0:
        raise FileNotFoundError("No .zip files found in tmp_zip/")
    if len(zips) > 1:
        raise RuntimeError(f"Multiple zip files found in tmp_zip/, please leave only one: {[p.name for p in zips]}")

    zip_to_extract = str(zips[0])
    extract_dir = tmp_dir / zips[0].stem

    # ensure a fresh extracted folder
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(zip_to_extract, "r") as z:
        z.extractall(str(extract_dir))
    # If the extracted folder contains nested zip files (sometimes exports are double-zipped),
    # extract them in-place and remove the nested zip files so downstream code sees the md/assets.
    nested_zips = list(extract_dir.rglob('*.zip'))
    for nz in nested_zips:
        try:
            with zipfile.ZipFile(str(nz), 'r') as nzf:
                nzf.extractall(str(extract_dir))
        except zipfile.BadZipFile:
            # ignore bad nested zips
            continue
        # remove the nested zip after extraction
        try:
            nz.unlink()
        except Exception:
            pass

    return extract_dir


def find_md_and_assets(folder: Path):
    """Find the first .md file and the first directory in the root of the extracted folder.

    Assumes Notion export contains one markdown file and one assets folder at the root.
    """
    md_files = list(folder.glob("*.md"))
    if not md_files:
        # also try to look one level deeper
        for sub in folder.iterdir():
            if sub.is_dir():
                md_files = list(sub.glob("*.md"))
                if md_files:
                    assets_folder = next((p for p in folder.iterdir() if p.is_dir()), None)
                    return md_files[0], assets_folder
        raise FileNotFoundError("No .md file found in Notion export (root or one level deeper).")
    md_file = md_files[0]
    assets_folder = next((p for p in folder.iterdir() if p.is_dir()), None)
    return md_file, assets_folder


def sanitize_name(name: str) -> str:
    """Remove trailing id-like suffix and replace spaces with underscores.

    Example: 'Attention is All You Need 176559a0...' -> 'Attention_is_All_You_Need'
    """
    # remove extension if present
    base = name
    # strip trailing hex/id-ish suffix preceded by space
    base = re.sub(r"\s+[0-9a-fA-F\-]+$", "", base)
    # collapse whitespace
    base = re.sub(r"\s+", " ", base).strip()
    slug = base.replace(' ', '_')
    return slug


def rename_files_replace_spaces(folder: Path):
    """Rename files in folder (non-recursive) replacing spaces with underscores."""
    for p in folder.iterdir():
        if p.is_file():
            if ' ' in p.name:
                new_name = p.name.replace(' ', '_')
                dest = p.with_name(new_name)
                p.rename(dest)
        elif p.is_dir():
            # recurse into subfolders
            rename_files_replace_spaces(p)


def fix_markdown_image_links(md_text: str) -> str:
    """Normalize image links so they point to the local filename in the bundle.

    Only transform image links (markdown image syntax `![alt](url)`) and links
    that reference an `assets/` path. Do NOT touch regular external links.
    Examples:
    - Convert `![alt](Attention%20is%20All%20You%20Need/.../image 1.png)` -> `![alt](image_1.png)`
    - Leave `[Page Source](https://arxiv.org/...)` unchanged.
    """

    # First, handle explicit image syntax: ![alt](url)
    def repl_image(match):
        alt = match.group(1)
        inner = match.group(2).strip()
        # If it's a URL with percent-encoding, decode it first
        try:
            decoded = unquote(inner)
        except Exception:
            decoded = inner
        # take basename only (drop any folder components)
        base = os.path.basename(decoded)
        # replace spaces with underscores
        base = base.replace(' ', '_')
        return f'![{alt}]({base})'

    md_text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', repl_image, md_text)

    # Then handle any explicit occurrences of assets/... in normal links
    def repl_assets_link(match):
        inner = match.group(1)
        try:
            decoded = unquote(inner)
        except Exception:
            decoded = inner
        # Skip external URLs (http, https, protocol-relative) to avoid mangling them
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', decoded) or decoded.startswith('//'):
            return match.group(0)
        # Only rewrite when it looks like a local asset path or contains path components
        if 'assets/' in decoded or decoded.startswith('./') or os.path.dirname(decoded):
            base = os.path.basename(decoded).replace(' ', '_')
            return f'({base})'
        return match.group(0)

    md_text = re.sub(r'\(([^)]+)\)', repl_assets_link, md_text)
    return md_text


def create_hugo_bundle(md_file: Path, assets_folder: Path, description: str) -> Path:
    # 1) compute slug from file name
    raw_name = md_file.stem
    slug = sanitize_name(raw_name)

    dest_folder = Path(f"content/posts/{slug}")
    dest_folder.mkdir(parents=True, exist_ok=True)

    # 2) process assets: rename files to remove spaces, then move into dest_folder
    if assets_folder and assets_folder.exists():
        rename_files_replace_spaces(assets_folder)
        # move every file (and files in subfolders) to dest_folder root
        for p in assets_folder.rglob('*'):
            if p.is_file():
                target = dest_folder / p.name
                # if file already exists, add a numeric suffix to avoid clobbering
                if target.exists():
                    base, ext = os.path.splitext(p.name)
                    i = 1
                    while True:
                        candidate = dest_folder / f"{base}_{i}{ext}"
                        if not candidate.exists():
                            target = candidate
                            break
                        i += 1
                shutil.copy2(p, target)

    # 3) read markdown, drop first title line if it's '# ...'
    with open(md_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if lines and re.match(r'^#\s+', lines[0]):
        lines = lines[1:]

    md_text = ''.join(lines)
    md_text = fix_markdown_image_links(md_text)

    # 4) write index.md with front matter
    # prepare title from slug (replace underscores with spaces)
    title = slug.replace('_', ' ')
    # if no description provided, default to 'Paper-reading notes: {title}'
    if description and description.strip():
        final_desc = description.strip()
    else:
        final_desc = f"Paper-reading notes: {title}"

    # escape single quotes in YAML single-quoted strings by doubling them
    esc_title = title.replace("'", "''")
    esc_desc = final_desc.replace("'", "''")

    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()

    front = (
        '---\n'
        f"title: '{esc_title}'\n"
        f"date: {now}\n"
        "draft: false\n"
        f"description: '{esc_desc}'\n"
        "ShowWordCount: true\n"
        "ShowReadingTime: false\n"
        '---\n\n'
    )

    with open(dest_folder / 'index.md', 'w', encoding='utf-8') as f:
        f.write(front)
        f.write(md_text)

    return dest_folder


def main(description: str = ''):
    extracted = unzip_notion()
    md_file, assets_folder = find_md_and_assets(extracted)
    dest = create_hugo_bundle(md_file, assets_folder, description)
    print('Created Hugo bundle at', dest)


if __name__ == '__main__':
    import sys

    desc = sys.argv[2] if len(sys.argv) > 2 else ''
    main(desc)
