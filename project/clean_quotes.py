#!/usr/bin/env python3
"""
Clean quotes.csv and write a new file quotes_cleaned.csv.
Cleaning steps:
 - Normalize smart quotes and apostrophes
 - Strip surrounding quotes and whitespace
 - Collapse repeated whitespace and remove line breaks
 - Normalize tags (lowercase, stripped, comma-separated)
 - Fill missing authors with 'Unknown'
 - Remove exact duplicate quotes (keeping first occurrence)
"""

import csv
import re
from pathlib import Path

INPUT = Path("quotes.csv")
OUTPUT = Path("quotes_cleaned.csv")

SMART_QUOTES_MAP = {
    '\u201c': '"',  # left double
    '\u201d': '"',  # right double
    '\u2018': "'",  # left single
    '\u2019': "'",  # right single / apostrophe
    '\u2014': '-',
    '\u2013': '-',
}

RE_MULTISPACE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    # replace unicode smart quotes
    for k, v in SMART_QUOTES_MAP.items():
        s = s.replace(k, v)
    # replace special left/right double quotes still in ASCII-like forms
    s = s.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    # collapse whitespace including newlines
    s = RE_MULTISPACE.sub(' ', s)
    # strip surrounding whitespace
    s = s.strip()
    # strip surrounding " if entire field is wrapped in extra quotes
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()
    # also remove repeated outer quotes like a leading and trailing smart quote
    if len(s) >= 2 and (s[0] == '"' or s[0] == '“') and (s[-1] == '"' or s[-1] == '”'):
        s = s[1:-1].strip()
    return s


def clean_tags(tag_field: str) -> str:
    if not tag_field:
        return ''
    # normalize quotes and whitespace
    tag_field = normalize_text(tag_field)
    # split on comma
    parts = [p.strip().lower() for p in tag_field.split(',')]
    parts = [p for p in parts if p]
    return ','.join(parts)


def main():
    if not INPUT.exists():
        print(f"Input file {INPUT} not found.")
        return

    seen_quotes = set()
    rows_out = []
    stats = {
        'in': 0,
        'out': 0,
        'duplicates': 0,
        'missing_author': 0,
    }

    with INPUT.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats['in'] += 1
            raw_quote = row.get('quote') or ''
            raw_author = row.get('author') or ''
            raw_tags = row.get('tags') or ''

            quote = normalize_text(raw_quote)
            author = normalize_text(raw_author)
            tags = clean_tags(raw_tags)

            if not author:
                author = 'Unknown'
                stats['missing_author'] += 1

            norm_key = quote.lower()
            if norm_key in seen_quotes:
                stats['duplicates'] += 1
                continue
            seen_quotes.add(norm_key)

            rows_out.append({'quote': quote, 'author': author, 'tags': tags})

    stats['out'] = len(rows_out)

    # write output CSV
    with OUTPUT.open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['quote', 'author', 'tags']
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    print('Cleaning complete.')
    print(f"Rows read: {stats['in']}")
    print(f"Rows written (unique): {stats['out']}")
    print(f"Duplicates removed: {stats['duplicates']}")
    print(f"Missing authors filled: {stats['missing_author']}")
    print(f"Output file: {OUTPUT.resolve()}")

    # show a short preview
    print('\nPreview (first 10 rows):')
    for r in rows_out[:10]:
        print('-', r['quote'][:120].replace('\n',' '), '—', r['author'], '| tags:', r['tags'])


if __name__ == '__main__':
    main()
