
# Wafer Defect Pattern Detection

An algorithmic project for identifying and analyzing defect patterns in semiconductor wafers using defect coordinates and statistical techniques.

## Overview

Wafer Defect Pattern Detection processes wafer defect data to identify spatial patterns and analyze defect trends across manufacturing lots and processes. The project is organized into five milestones, covering pattern detection, trend analysis, and association analysis.

## Defect Patterns

The implementation includes detection or analysis of the following patterns:

- **Ring** — defects concentrated in a circular band within the wafer.
- **EdgeRing** — defects concentrated near the wafer's outer region.
- **Cluster** — defects concentrated within a localized area.
- **Scratch** — defects arranged in a line-like or elongated pattern.

## Project Structure

```text
wafer_defect_pattern_detection/
├── README.md
├── requirements.txt
├── Milestone Answers/
│   ├── m1 answers/
│   ├── m2 answers/
│   ├── m3 answers/
│   ├── m4 answers/
│   └── m5 answers/
├── Milestones/
│   ├── milestones_1_2_3.py
│   ├── milestone_4.py
│   └── milestone_5.py
├── sample_data/
└── src/
```

*The folder names and file locations above should match your actual repository.*

## Techniques Used

- Geometric calculations and radial statistics
- Angular sectors and wafer-coordinate analysis
- Principal Component Analysis (PCA) for shape analysis
- Grid-based spatial neighborhood search
- Breadth-First Search (BFS) for connected-component detection
- Statistical testing and multiple-testing correction
- Graph-based association analysis using the Bron–Kerbosch algorithm

## Milestones

- **Milestone 1:** Initial wafer defect pattern detection.
- **Milestone 2:** Additional pattern detection and classification.
- **Milestone 3:** Spatial pattern analysis and connected-component processing.
- **Milestone 4:** Lot- and process-level recurrence and trend analysis.
- **Milestone 5:** Association analysis between defect patterns.

Refer to the corresponding Python files and the `Milestone Answers` directory for implementation details and outputs.

## Installation

### Prerequisites

- Python 3.10 or a compatible Python version
- pip

### Setup

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd wafer_defect_pattern_detection
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Input Data

The implementation works with wafer defect coordinate data, including metadata such as wafer ID, lot ID, process ID, process sequence, and defect coordinates.

Use the input format and file paths expected by the corresponding milestone script. Add only sample data that you are permitted to share publicly.

## Running the Code

Run a milestone script from the repository root using the appropriate input file and command-line arguments, if required.

For example:

```bash
python "Milestones/milestones_1_2_3.py"
```

**Note:** Update the command and input/output paths to match the actual script configuration before running. Not every milestone script necessarily uses the same input format or arguments.

## Dependencies

The `requirements.txt` file lists the external Python packages needed by the implementation. Python standard-library modules do not need to be installed separately.

## Disclaimer

This repository documents a wafer defect pattern detection project. Detection accuracy and execution depend on the supplied data, implementation, and configuration.

Check the applicable hackathon rules before publishing solution code, answer files, or test cases.

## Author

Developed as a semiconductor wafer defect pattern detection project.

# wafer-defect-pattern-detection
