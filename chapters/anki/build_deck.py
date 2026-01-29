#!/usr/bin/env python3
"""
Build Anki deck (.apkg) from CSV files and images.

Usage:
    python build_deck.py                    # Build complete deck
    python build_deck.py --output my.apkg   # Custom output name
    python build_deck.py --chapters 1 2 3   # Specific chapters only
    python build_deck.py --no-images        # Skip images

Requirements:
    pip install genanki

Output:
    Creates .apkg file that can be imported directly into Anki with all
    cards and images bundled together.
"""

import os
import csv
import glob
import argparse
import random
import hashlib
from pathlib import Path

try:
    import genanki
except ImportError:
    print("ERROR: genanki not installed. Run: pip install genanki")
    exit(1)

# Paths
SCRIPT_DIR = Path(__file__).parent
CSV_DIR = SCRIPT_DIR
IMAGES_DIR = SCRIPT_DIR / 'images'
DEFAULT_OUTPUT = SCRIPT_DIR / 'ml_flashcards.apkg'

# Generate consistent IDs based on deck name
def generate_id(name: str) -> int:
    """Generate a consistent ID from a string."""
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)

DECK_ID = generate_id('Python ML Flashcards')
BASIC_MODEL_ID = generate_id('Basic ML Model')
CLOZE_MODEL_ID = generate_id('Cloze ML Model')

# Card styling
CARD_CSS = """
.card {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    font-size: 18px;
    text-align: center;
    color: #333;
    background-color: #fafafa;
    padding: 20px;
    line-height: 1.5;
}

.card img {
    max-width: 100%;
    max-height: 400px;
    margin: 10px auto;
    display: block;
}

code {
    font-family: 'SF Mono', 'Fira Code', Consolas, 'Courier New', monospace;
    font-size: 16px;
    background-color: #f0f0f0;
    padding: 2px 6px;
    border-radius: 4px;
    color: #d63384;
}

pre {
    font-family: 'SF Mono', 'Fira Code', Consolas, 'Courier New', monospace;
    font-size: 14px;
    background-color: #2d2d2d;
    color: #f8f8f2;
    padding: 12px;
    border-radius: 6px;
    text-align: left;
    overflow-x: auto;
    white-space: pre-wrap;
}

.tags {
    font-size: 12px;
    color: #888;
    margin-top: 15px;
}

.cloze {
    font-weight: bold;
    color: #0066cc;
}

.front {
    font-size: 20px;
    font-weight: 500;
}

.back {
    font-size: 18px;
}
"""

# Basic card model
BASIC_MODEL = genanki.Model(
    BASIC_MODEL_ID,
    'ML Basic',
    fields=[
        {'name': 'Front'},
        {'name': 'Back'},
        {'name': 'Tags'},
    ],
    templates=[
        {
            'name': 'Card 1',
            'qfmt': '<div class="front">{{Front}}</div>',
            'afmt': '''
                <div class="front">{{Front}}</div>
                <hr id="answer">
                <div class="back">{{Back}}</div>
                <div class="tags">{{Tags}}</div>
            ''',
        },
    ],
    css=CARD_CSS,
)

# Cloze card model
CLOZE_MODEL = genanki.Model(
    CLOZE_MODEL_ID,
    'ML Cloze',
    model_type=genanki.Model.CLOZE,
    fields=[
        {'name': 'Text'},
        {'name': 'Extra'},
        {'name': 'Tags'},
    ],
    templates=[
        {
            'name': 'Cloze',
            'qfmt': '<div class="front">{{cloze:Text}}</div>',
            'afmt': '''
                <div class="front">{{cloze:Text}}</div>
                <div class="back">{{Extra}}</div>
                <div class="tags">{{Tags}}</div>
            ''',
        },
    ],
    css=CARD_CSS,
)


def format_code(text: str) -> str:
    """Format code snippets in text."""
    # Simple code detection and formatting
    # If the answer looks like code (has common code patterns), wrap in <code>
    code_indicators = ['()', '.', '=', '[', ']', 'import ', 'from ', 'def ', 'class ']

    if any(ind in text for ind in code_indicators) and not text.startswith('<'):
        # Check if it's a multi-statement (has ;)
        if ';' in text or '\n' in text:
            return f'<pre>{text}</pre>'
        else:
            return f'<code>{text}</code>'
    return text


def parse_csv(filepath: Path) -> list:
    """Parse a CSV file and return list of cards."""
    cards = []

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')

        for row in reader:
            card_type = (row.get('Type') or '').strip()
            front = (row.get('Front') or '').strip()
            back = (row.get('Back') or '').strip()
            tags = (row.get('Tags') or '').strip()

            if not front:
                continue

            # Format code in answers
            if card_type == 'Basic':
                back = format_code(back)

            cards.append({
                'type': card_type,
                'front': front,
                'back': back,
                'tags': tags,
            })

    return cards


def get_chapter_number(filename: str) -> int:
    """Extract chapter number from filename."""
    # chapter_01_anki.csv -> 1
    try:
        parts = filename.split('_')
        return int(parts[1])
    except (IndexError, ValueError):
        return 99


def build_deck(
    output_path: Path = DEFAULT_OUTPUT,
    chapters: list = None,
    include_images: bool = True,
) -> None:
    """Build the Anki deck from CSV files."""

    # Create deck
    deck = genanki.Deck(
        DECK_ID,
        'Python Machine Learning Flashcards'
    )

    # Find CSV files
    csv_files = sorted(
        glob.glob(str(CSV_DIR / 'chapter_*_anki.csv')),
        key=lambda x: get_chapter_number(Path(x).name)
    )

    if not csv_files:
        print(f"ERROR: No CSV files found in {CSV_DIR}")
        return

    # Filter chapters if specified
    if chapters:
        csv_files = [
            f for f in csv_files
            if get_chapter_number(Path(f).name) in chapters
        ]

    print(f"Building deck from {len(csv_files)} CSV files...\n")

    total_cards = 0
    media_files = []

    for csv_file in csv_files:
        filename = Path(csv_file).name
        chapter_num = get_chapter_number(filename)
        cards = parse_csv(Path(csv_file))

        print(f"  Chapter {chapter_num:02d}: {len(cards)} cards")

        for card in cards:
            # Create appropriate note type
            if card['type'] == 'Cloze':
                note = genanki.Note(
                    model=CLOZE_MODEL,
                    fields=[card['front'], card['back'], card['tags']],
                    tags=card['tags'].replace('::', '_').split() if card['tags'] else [],
                )
            else:
                note = genanki.Note(
                    model=BASIC_MODEL,
                    fields=[card['front'], card['back'], card['tags']],
                    tags=card['tags'].replace('::', '_').split() if card['tags'] else [],
                )

            deck.add_note(note)
            total_cards += 1

    # Collect images
    if include_images and IMAGES_DIR.exists():
        image_files = list(IMAGES_DIR.glob('*.png')) + list(IMAGES_DIR.glob('*.jpg'))
        media_files = [str(f) for f in image_files]
        if media_files:
            print(f"\n  Including {len(media_files)} images")

    # Create package
    package = genanki.Package(deck)
    if media_files:
        package.media_files = media_files

    # Write output
    package.write_to_file(str(output_path))

    print(f"\n{'=' * 50}")
    print(f"SUCCESS: Created {output_path.name}")
    print(f"  Total cards: {total_cards}")
    print(f"  Media files: {len(media_files)}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"\nImport this file into Anki to use the deck.")


def main():
    parser = argparse.ArgumentParser(
        description='Build Anki deck from CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_deck.py                      # Build complete deck
  python build_deck.py --output study.apkg  # Custom filename
  python build_deck.py --chapters 1 2 3     # Only chapters 1, 2, 3
  python build_deck.py --no-images          # Exclude images
        """
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=DEFAULT_OUTPUT,
        help='Output .apkg file path'
    )

    parser.add_argument(
        '--chapters', '-c',
        type=int,
        nargs='+',
        help='Specific chapter numbers to include'
    )

    parser.add_argument(
        '--no-images',
        action='store_true',
        help='Exclude image files from package'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available CSV files and exit'
    )

    args = parser.parse_args()

    if args.list:
        csv_files = sorted(glob.glob(str(CSV_DIR / 'chapter_*_anki.csv')))
        print("Available CSV files:\n")
        for f in csv_files:
            cards = parse_csv(Path(f))
            print(f"  {Path(f).name}: {len(cards)} cards")
        print(f"\nTotal: {len(csv_files)} files")
        return

    build_deck(
        output_path=args.output,
        chapters=args.chapters,
        include_images=not args.no_images,
    )


if __name__ == '__main__':
    main()
