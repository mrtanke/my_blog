import os
import re
from urllib.parse import unquote, urlparse
import zipfile
import shutil
import datetime
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
    """Find the markdown file and its related assets folder in extracted Notion exports.

    Supports older root-level exports and newer nested structures like:
    - <extract-root>/ExportBlock-...-Part-1.zip
      -> Private & Shared/<title with uuid>.md
      -> Private & Shared/<title>/<asset files>
    """

    def _title_without_uuid(name: str) -> str:
        return re.sub(r"\s+[0-9a-fA-F\-]{8,}$", "", name).strip()

    md_files = sorted(p for p in folder.rglob("*.md") if p.is_file())
    if not md_files:
        raise FileNotFoundError("No .md file found in extracted Notion export.")

    md_file = md_files[0]
    parent = md_file.parent
    stem = md_file.stem
    plain_title = _title_without_uuid(stem)

    candidate_dirs: list[Path] = []
    exact_stem_dir = parent / stem
    plain_title_dir = parent / plain_title
    if exact_stem_dir.is_dir():
        candidate_dirs.append(exact_stem_dir)
    if plain_title_dir.is_dir() and plain_title_dir not in candidate_dirs:
        candidate_dirs.append(plain_title_dir)

    # Some exports include slight naming differences in the asset directory.
    for sibling in sorted(parent.iterdir()):
        if not sibling.is_dir():
            continue
        if sibling in candidate_dirs:
            continue
        if sibling.name.startswith(plain_title) or plain_title.startswith(sibling.name):
            candidate_dirs.append(sibling)

    assets_folder = candidate_dirs[0] if candidate_dirs else None
    return md_file, assets_folder


def sanitize_name(name: str) -> str:
    """Convert a Notion-exported title into a lowercase, hyphenated slug."""
    base = re.sub(r"\s+[0-9a-fA-F\-]+$", "", name)
    base = re.sub(r"\s+", " ", base).strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", base)
    slug = re.sub(r"-+", "-", slug).strip('-')
    return slug or "post"


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

    # Protect math blocks ($$...$$, $...$, \[...\], \(...\)) so we don't rewrite
    # parentheses that are part of LaTeX expressions.
    math_patterns = re.compile(r'(\$\$.*?\$\$)|(\$.*?\$)|(\\\[.*?\\\])|(\\\(.*?\\\))', re.DOTALL)
    math_blocks: list[str] = []

    def _protect_math(m):
        math_blocks.append(m.group(0))
        return f"@@MATH{len(math_blocks)-1}@@"

    protected = math_patterns.sub(_protect_math, md_text)

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

    protected = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', repl_image, protected)

    # Then handle markdown links `[text](assets/...)` while leaving external URLs untouched.
    def repl_assets_link(match):
        text = match.group(1)
        inner = match.group(2).strip()
        try:
            decoded = unquote(inner)
        except Exception:
            decoded = inner
        parsed = urlparse(decoded)
        if parsed.scheme or parsed.netloc or decoded.startswith('//'):
            return match.group(0)
        if decoded.startswith('assets/') or decoded.startswith('./') or decoded.startswith('../') or os.path.dirname(decoded):
            base = os.path.basename(decoded).replace(' ', '_')
            return f'[{text}]({base})'
        return match.group(0)

    protected = re.sub(r'(?<!!)\[([^\]]+)\]\(([^)]+)\)', repl_assets_link, protected)

    # restore math blocks
    def _restore_math(match):
        idx = int(match.group(1))
        return math_blocks[idx]

    protected = re.sub(r'@@MATH(\d+)@@', _restore_math, protected)
    return protected


def collapse_duplicate_image_captions(md_text: str) -> str:
    """If image alt text is repeated on the next non-empty line, fold it into image title.

    Example:
    - ![Caption](img.png)\n\nCaption
    becomes:
    - ![Caption](img.png "Caption")
    """

    image_line = re.compile(r'^(\s*)!\[([^\]]*)\]\((\S+?)(?:\s+"([^"]*)")?\)\s*$')
    lines = md_text.splitlines(keepends=True)
    i = 0

    while i < len(lines):
        current = lines[i].rstrip('\r\n')
        m = image_line.match(current)
        if not m:
            i += 1
            continue

        indent, alt, url, title = m.group(1), m.group(2).strip(), m.group(3), m.group(4)

        if title:
            i += 1
            continue

        j = i + 1
        while j < len(lines) and lines[j].strip() == '':
            j += 1

        if j >= len(lines):
            i += 1
            continue

        next_text = lines[j].strip()
        if alt and next_text == alt:
            line_ending = '\n' if lines[i].endswith('\n') else ''
            safe_title = alt.replace('"', '\\"')
            lines[i] = f'{indent}![{alt}]({url} "{safe_title}"){line_ending}'
            del lines[j]

        i += 1

    return ''.join(lines)


def create_hugo_bundle(
    md_file: Path,
    assets_folder: Path,
    description: str,
    tag: str,
    tags: list[str] | None = None,
) -> Path:
    # 1) compute slug from file name
    raw_name = md_file.stem
    slug = sanitize_name(raw_name)
    date_prefix = datetime.date.today().isoformat()
    bundle_name = f"{date_prefix}-{slug}"

    # decide destination base based on tag so generated posts go to the matching
    # top-level folder (e.g. content/posts, content/projects).
    tag_key = (tag or '').strip().lower()
    tag_map = {
        'posts': 'posts', 'post': 'posts',
        'projects': 'projects', 'project': 'projects',
    }
    base = tag_map.get(tag_key, 'posts')
    dest_folder = Path(f"content/{base}/{bundle_name}")
    counter = 1
    while dest_folder.exists():
        dest_folder = Path(f"content/{base}/{bundle_name}-{counter}")
        counter += 1
    dest_folder.mkdir(parents=True, exist_ok=False)

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
    md_text = collapse_duplicate_image_captions(md_text)

    # 4) write index.md with front matter
    # prepare title from slug (replace hyphens with spaces) and normalize to title case
    title = slug.replace('-', ' ').title()
    # if no description provided, default to 'Paper-reading notes: {title}'
    if description and description.strip():
        final_desc = description.strip()
    else:
        final_desc = f"Paper-reading notes: {title}"

    # escape single quotes in YAML single-quoted strings by doubling them
    esc_title = title.replace("'", "''")
    esc_desc = final_desc.replace("'", "''")

    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()

    esc_tag = tag.replace("'", "''") if tag else 'Posts'
    cleaned_tags = []
    for item in tags or []:
        item = item.strip()
        if item:
            cleaned_tags.append(item)
    if not cleaned_tags:
        cleaned_tags = [tag or 'Posts']
    escaped_tags = [t.replace("'", "''") for t in cleaned_tags]
    tags_block = "tags:\n" + ''.join(f"  - '{t}'\n" for t in escaped_tags)
    front = (
        '---\n'
        f"title: '{esc_title}'\n"
        f"date: {now}\n"
        "draft: false\n"
        f"description: '{esc_desc}'\n"
        f"tag: '{esc_tag}'\n"
        "ShowWordCount: true\n"
        "ShowReadingTime: false\n"
        f"{tags_block}"
        '---\n\n'
    )

    with open(dest_folder / 'index.md', 'w', encoding='utf-8') as f:
        f.write(front)
        f.write(md_text)

    return dest_folder


def main(description: str = '', tag: str = 'Posts', tags: list[str] | None = None):
    extracted = unzip_notion()
    md_file, assets_folder = find_md_and_assets(extracted)
    dest = create_hugo_bundle(md_file, assets_folder, description, tag, tags)
    print('Created Hugo bundle at', dest)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Unzip Notion export and create Hugo post bundle')
    p.add_argument('--description', nargs='?', default='', help='optional description to use in front matter')
    p.add_argument('--tag', dest='tag', default='Posts', help='Section selector (Posts, Projects) used for the bundle path and fallback tag')
    p.add_argument('--tags', nargs='*', default=None, help='Space- or comma-separated tags to include in front matter (e.g. --tags vision-model rag)')
    args = p.parse_args()

    cli_tags: list[str] | None = None
    if args.tags:
        parsed: list[str] = []
        for entry in args.tags:
            parts = [part.strip() for part in entry.split(',') if part.strip()]
            parsed.extend(parts)
        cli_tags = parsed or None

    main(args.description, args.tag, cli_tags)
