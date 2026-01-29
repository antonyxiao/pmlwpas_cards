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
import re
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

# Card styling - Modern dark mode design
CARD_CSS = """
.card {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    font-size: 18px;
    text-align: center;
    color: #e4e4e7;
    background: linear-gradient(145deg, #18181b 0%, #1f1f23 100%);
    padding: 32px 24px;
    line-height: 1.6;
    min-height: 100vh;
    box-sizing: border-box;
}

.card img {
    max-width: 100%;
    max-height: 400px;
    margin: 16px auto;
    display: block;
    border-radius: 8px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
}

code {
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', Consolas, monospace;
    font-size: 0.9em;
    background-color: #27272a;
    padding: 3px 8px;
    border-radius: 6px;
    color: #a78bfa;
    border: 1px solid #3f3f46;
}

pre {
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', Consolas, monospace;
    font-size: 14px;
    background-color: #0f0f11;
    color: #e4e4e7;
    padding: 16px 20px;
    border-radius: 12px;
    text-align: left;
    overflow-x: auto;
    white-space: pre-wrap;
    border: 1px solid #27272a;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
    margin: 12px 0;
}

hr#answer {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #3f3f46, transparent);
    margin: 24px 0;
}

.tags {
    font-size: 11px;
    color: #71717a;
    margin-top: 24px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.cloze {
    font-weight: 600;
    color: #60a5fa;
    background-color: rgba(96, 165, 250, 0.15);
    padding: 2px 6px;
    border-radius: 4px;
}

.front {
    font-size: 22px;
    font-weight: 500;
    color: #fafafa;
    line-height: 1.5;
}

.back {
    font-size: 18px;
    color: #a1a1aa;
    line-height: 1.6;
}

/* Highlight key terms */
strong, b {
    color: #f472b6;
    font-weight: 600;
}

em, i {
    color: #67e8f9;
    font-style: italic;
}
"""

# Basic card model
BASIC_MODEL = genanki.Model(
    BASIC_MODEL_ID,
    'ML Basic',
    fields=[
        {'name': 'SortOrder'},  # Hidden field for ordering
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
    sort_field_index=0,  # Sort by SortOrder field
)

# Cloze card model
CLOZE_MODEL = genanki.Model(
    CLOZE_MODEL_ID,
    'ML Cloze',
    model_type=genanki.Model.CLOZE,
    fields=[
        {'name': 'SortOrder'},  # Hidden field for ordering
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
    sort_field_index=0,  # Sort by SortOrder field
)


def format_code(text: str) -> str:
    """Format code snippets in text - conservative detection."""
    # Skip if already has HTML tags
    if text.startswith('<') or '<code>' in text or '<pre>' in text:
        return text

    # Strong code indicators - definitely code
    strong_indicators = ['import ', 'from ', 'def ', 'class ', ' = ', '()', '[]',
                         '.fit(', '.predict(', '.transform(', 'print(', 'return ',
                         'if ', 'for ', 'while ', 'lambda ', '.__', '::']

    # Check for strong indicators
    has_strong = any(ind in text for ind in strong_indicators)

    # Additional check: looks like a function call (word followed by parentheses with content)
    func_call = re.search(r'\w+\([^)]*\)', text)

    # Only format if it has strong indicators AND looks like actual code
    if has_strong and func_call:
        # Multi-line or multiple statements
        if '\n' in text or '; ' in text:
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


# Chapter titles for subdeck names
CHAPTER_TITLES = {
    1: "ML Foundations",
    2: "Basic Algorithms",
    3: "Scikit-Learn Classifiers",
    4: "Data Preprocessing",
    5: "Dimensionality Reduction",
    6: "Model Evaluation",
    7: "Ensemble Learning",
    8: "Sentiment Analysis",
    9: "Regression",
    10: "Clustering",
    11: "Neural Networks from Scratch",
    12: "PyTorch Basics",
    13: "PyTorch Advanced",
    14: "CNNs",
    15: "RNNs",
    16: "Transformers",
    17: "GANs",
    18: "Graph Neural Networks",
    19: "Reinforcement Learning",
}


def build_deck(
    output_path: Path = DEFAULT_OUTPUT,
    chapters: list = None,
    include_images: bool = True,
) -> None:
    """Build the Anki deck from CSV files with chapter subdecks."""

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

    # Create subdecks for each chapter
    decks = []
    total_cards = 0
    media_files = []

    for csv_file in csv_files:
        filename = Path(csv_file).name
        chapter_num = get_chapter_number(filename)
        cards = parse_csv(Path(csv_file))

        # Create subdeck with chapter name
        chapter_title = CHAPTER_TITLES.get(chapter_num, f"Chapter {chapter_num}")
        subdeck_name = f"Python ML Flashcards::Ch{chapter_num:02d} - {chapter_title}"
        subdeck_id = generate_id(subdeck_name)

        subdeck = genanki.Deck(subdeck_id, subdeck_name)

        print(f"  Chapter {chapter_num:02d}: {len(cards)} cards")

        for card_index, card in enumerate(cards, start=1):
            # Create sort order: "CC-NNN" (chapter-cardnum) for proper ordering
            sort_order = f"{chapter_num:02d}-{card_index:04d}"

            # Create appropriate note type
            if card['type'] == 'Cloze':
                note = genanki.Note(
                    model=CLOZE_MODEL,
                    fields=[sort_order, card['front'], card['back'], card['tags']],
                    tags=card['tags'].replace('::', '_').split() if card['tags'] else [],
                )
            else:
                note = genanki.Note(
                    model=BASIC_MODEL,
                    fields=[sort_order, card['front'], card['back'], card['tags']],
                    tags=card['tags'].replace('::', '_').split() if card['tags'] else [],
                )

            subdeck.add_note(note)
            total_cards += 1

        decks.append(subdeck)

    # Collect images
    if include_images and IMAGES_DIR.exists():
        image_files = list(IMAGES_DIR.glob('*.png')) + list(IMAGES_DIR.glob('*.jpg'))
        media_files = [str(f) for f in image_files]
        if media_files:
            print(f"\n  Including {len(media_files)} images")

    # Create package with all subdecks
    package = genanki.Package(decks)
    if media_files:
        package.media_files = media_files

    # Write output
    package.write_to_file(str(output_path))

    print(f"\n{'=' * 50}")
    print(f"SUCCESS: Created {output_path.name}")
    print(f"  Total cards: {total_cards}")
    print(f"  Subdecks: {len(decks)}")
    print(f"  Media files: {len(media_files)}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"\nImport this file into Anki to use the deck.")
    print(f"Cards are organized by chapter in the deck browser.")


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
