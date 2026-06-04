# Harmonic Entrepreneur Analysis Framework

This repository contains the code, dataset, and analysis pipeline used to evaluate the **Harmonic Entrepreneur (HE) Framework** through survey-based data collected at the University of Calabria.

The framework assesses multiple dimensions of entrepreneurship, including well-being, ethics, innovation, stakeholder relationships, social responsibility, and self-awareness.

## Repository Structure

```text
Harmonic_Entrep/
├── Survey Dataset/
│   └── data_unical_2026-05-12_15-12.xlsx
├── scripts/
│   ├── harmonic_entrepreneur_analysis.py
│   ├── classifiacation_methodology.py
│   └── classification_analysis.py
├── results/
└── Chapter 4 Harmonic.pdf
```

## Main Components

### harmonic_entrepreneur_analysis.py
Performs data preprocessing, HE score computation, reliability analysis, demographic analysis, and visualization.

### classifiacation_methodology.py
Classifies participants into learning paradigms based on textual response characteristics.

### classification_analysis.py
Performs statistical comparisons and generates visualizations for the identified participant groups.

## Installation

```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn openpyxl pingouin
```

## Usage

Run the analysis pipeline in the following order:

```bash
python harmonic_entrepreneur_analysis.py
python classifiacation_methodology.py
python classification_analysis.py
```

Generated outputs, figures, and statistical results will be stored in the `results/` directory.

## Harmonic_chapter.pdf 

This pdf is the Harmonic Textbook used for the survey.

## Dataset

The repository includes the survey dataset used in the study, containing:

- Demographic information
- Likert-scale responses
- Open-ended responses
- Response timing information

