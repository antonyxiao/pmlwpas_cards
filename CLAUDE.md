# CLAUDE.md - AI Assistant Guide for pmlwpas_cards

This document provides guidance for AI assistants working with this repository.

## Project Overview

**pmlwpas_cards** is an educational resource repository supporting learning from the textbook "Python Machine Learning with Python Scikit-Learn." It contains:

- Split PDF chapters from the comprehensive machine learning textbook
- Anki flashcard study materials for spaced repetition learning
- Python utilities for PDF processing and Anki deck building

This is a **study/documentation repository**, not a production codebase.

## Repository Structure

```
pmlwpas_cards/
├── py_ml_w_py_skl.pdf              # Master PDF - complete textbook (~60 MB)
├── split_chapters.py               # Python utility for splitting PDFs
├── chapters/                       # Individual chapter PDFs
│   ├── Chapter_01_*.pdf            # Chapters 1-19 (machine learning curriculum)
│   ├── ...
│   └── Chapter_19_*.pdf
└── chapters/anki/                  # Anki flashcard materials
    ├── chapter_01_anki.csv         # Flashcards for each chapter (source)
    ├── chapter_02_anki.csv
    ├── ...
    ├── chapter_19_anki.csv
    ├── images/                     # Generated diagrams for cards
    │   ├── ch03_decision_boundary.png
    │   ├── ch11_mlp_architecture.png
    │   └── ...
    ├── generate_diagrams.py        # Script to generate ML diagrams
    ├── build_deck.py               # Script to build .apkg from CSVs + images
    └── ml_flashcards.apkg          # Output: importable Anki deck
```

## Build Workflow

### Step 1: Generate Diagrams

```bash
cd chapters/anki

# Generate all diagrams
python generate_diagrams.py

# Generate specific chapter diagrams
python generate_diagrams.py ch05

# List available diagrams
python generate_diagrams.py --list
```

**Requirements:** `pip install matplotlib numpy networkx`

### Step 2: Build Anki Deck

```bash
# Build complete deck with images
python build_deck.py

# Custom output name
python build_deck.py --output my_deck.apkg

# Specific chapters only
python build_deck.py --chapters 1 2 3

# List available CSVs
python build_deck.py --list
```

**Requirements:** `pip install genanki`

### Step 3: Import to Anki

1. Open Anki
2. File → Import
3. Select `ml_flashcards.apkg`
4. Cards and images are automatically imported

## Key Files

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

### generate_diagrams.py

Generates ML concept diagrams programmatically using matplotlib.

**Available diagrams:**
| Chapter | Diagrams |
|---------|----------|
| 03 | Decision boundary, SVM margin |
| 05 | PCA projection, variance explained |
| 06 | Confusion matrix, ROC curve, bias-variance |
| 10 | K-means steps, elbow method |
| 11 | MLP architecture, activation functions |
| 14 | Convolution, pooling |
| 15 | RNN unrolled, LSTM cell |
| 17 | GAN architecture |
| 19 | RL loop, Q-learning |

### build_deck.py

Compiles CSV files and images into a single `.apkg` file using `genanki`.

**Features:**
- Bundles all cards and images together
- Formats code snippets with syntax highlighting
- Supports Basic and Cloze card types
- Customizable CSS styling

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
4. Keep answers short (2-10 words for easy recall)
5. For Cloze cards, use `{{c1::}}`, `{{c2::}}` etc. for multiple deletions
6. Add `<img src="filename.png">` to reference diagrams

### When Adding Diagrams

1. Add diagram function to `generate_diagrams.py`
2. Register in the `DIAGRAMS` dictionary
3. Use consistent styling (colors from `COLORS` dict)
4. Run script to generate the PNG
5. Reference in CSV with `<img src="ch##_name.png">`

### Flashcard Quality Standards

- Questions should be specific and unambiguous
- Answers should be short and recallable (2-10 words ideal)
- Use Cloze for definitions, formulas, and key terms
- Use Basic for conceptual questions and code snippets
- Include prerequisite cards for terms used before being defined
- Add code cards for practical implementation knowledge

## Conventions

### File Naming

- PDF chapters: `Chapter_XX_Title_With_Underscores.pdf`
- Anki files: `chapter_XX_anki.csv` (lowercase)
- Scripts: `snake_case.py`
- Images: `chXX_descriptive_name.png`

### Card Tag Structure

```
chapterXX::topic-subtopic
```

Examples:
- `chapter01::ml-types`
- `chapter03::sklearn-api`
- `chapter11::activation-functions`
- `chapter14::code`

## Notes for AI Assistants

1. **This is not a production codebase** - No testing framework, CI/CD, or build system
2. **Binary files dominate** - PDFs are large; avoid unnecessary operations on them
3. **Study material focus** - Changes should enhance learning effectiveness
4. **Anki compatibility** - Ensure CSV exports remain importable to Anki
5. **Run build scripts** after modifying CSVs to regenerate the .apkg

## Quick Reference

| Item | Count |
|------|-------|
| Chapters | 19 |
| Total Flashcards | ~1,100 |
| Generated Diagrams | 18 |
| PDF Pages | 718 (core content) |
| Python Files | 3 |

## Git Information

- **Remote:** origin
- **Main content:** Single initial commit
- **Branch naming:** Feature branches prefixed with `claude/`
