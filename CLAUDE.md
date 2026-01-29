# CLAUDE.md - AI Assistant Guide for pmlwpas_cards

This document provides guidance for AI assistants working with this repository.

## Project Overview

**pmlwpas_cards** is an educational resource repository supporting learning from the textbook "Python Machine Learning with Python Scikit-Learn." It contains:

- Split PDF chapters from the comprehensive machine learning textbook
- Anki flashcard study materials for spaced repetition learning
- A Python utility script for PDF processing

This is a **study/documentation repository**, not a production codebase.

## Repository Structure

```
pmlwpas_cards/
├── py_ml_w_py_skl.pdf          # Master PDF - complete textbook (~60 MB)
├── split_chapters.py           # Python utility for splitting PDFs
├── chapters/                   # Individual chapter PDFs
│   ├── Chapter_01_*.pdf        # Chapters 1-19 (machine learning curriculum)
│   ├── Chapter_02_*.pdf
│   ├── ...
│   └── Chapter_19_*.pdf
└── chapters/anki/              # Anki flashcard CSV files
    ├── chapter_01_anki.csv     # Flashcards for each chapter
    ├── chapter_02_anki.csv
    ├── ...
    └── chapter_19_anki.csv
```

## Key Files

### split_chapters.py

Python script for splitting the master PDF into individual chapter files.

**Dependencies:** `pypdf` library

**Usage:** Modify the hardcoded paths and run:
```bash
python split_chapters.py
```

### Anki CSV Files

**Location:** `chapters/anki/`

**Format:** Tab-separated CSV with headers:
```
Type	Front	Back	Tags
```

**Card Types:**
- `Basic` - Standard question/answer cards
- `Cloze` - Fill-in-the-blank cards using `{{c1::text}}` syntax

**Tag Convention:** Hierarchical tags like `chapter01::ml-types`, `chapter03::sklearn-api`

## Technologies Referenced

The study materials cover:

- **Python** - Primary programming language
- **scikit-learn** - Machine learning library
- **PyTorch** - Deep learning framework
- **NumPy/Pandas** - Data manipulation
- **Neural Networks** - CNNs, RNNs, Transformers, GANs, GNNs

## Curriculum Structure

| Chapters | Topic Area |
|----------|------------|
| 1-2 | ML Foundations & Basic Algorithms |
| 3-6 | Scikit-Learn: Classifiers, Preprocessing, Evaluation |
| 7-10 | Ensemble Methods, NLP, Regression, Clustering |
| 11-16 | Deep Learning: Neural Networks, PyTorch, CNNs, RNNs, Transformers |
| 17-19 | Advanced: GANs, Graph Neural Networks, Reinforcement Learning |

## Development Guidelines

### When Modifying Flashcards

1. Maintain the tab-separated format
2. Use appropriate card types (`Basic` or `Cloze`)
3. Follow the existing tag hierarchy: `chapterXX::topic-subtopic`
4. Keep answers concise but complete
5. For Cloze cards, use `{{c1::}}`, `{{c2::}}` etc. for multiple deletions

### When Modifying the PDF Splitter

1. Update the `chapters` dictionary with correct page ranges (1-indexed)
2. Ensure meaningful chapter names that become valid filenames
3. Test with a small subset before full processing

### Adding New Content

- New flashcard files should follow naming: `chapter_XX_anki.csv`
- New chapter PDFs should follow naming: `Chapter_XX_Title_With_Underscores.pdf`
- Keep study materials organized within the `chapters/` directory structure

## Conventions

### File Naming

- PDF chapters: `Chapter_XX_Title_With_Underscores.pdf`
- Anki files: `chapter_XX_anki.csv` (lowercase)
- Scripts: `snake_case.py`

### Flashcard Quality Standards

- Questions should be specific and unambiguous
- Answers should be verifiable from the source material
- Use Cloze for definitions and key terms
- Use Basic for conceptual questions and comparisons

## Notes for AI Assistants

1. **This is not a production codebase** - No testing framework, CI/CD, or build system
2. **Binary files dominate** - PDFs are large; avoid unnecessary operations on them
3. **Study material focus** - Changes should enhance learning effectiveness
4. **Anki compatibility** - Ensure CSV exports remain importable to Anki
5. **Page numbers are 1-indexed** in the split_chapters.py configuration but converted to 0-indexed for PyPDF array access

## Quick Reference

| Item | Count |
|------|-------|
| Chapters | 19 |
| Total Flashcards | ~1,076 |
| PDF Pages | 718 (core content) |
| Python Files | 1 |

## Git Information

- **Remote:** origin
- **Main content:** Single initial commit
- **Branch naming:** Feature branches prefixed with `claude/`
