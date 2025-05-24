# Tools Directory

This directory contains various utilities, debugging tools, and documentation for the Im2Height project.

## Directory Structure

```
tools/
├── README.md                 # This file - overview of tools organization
├── debug/                    # Debugging and diagnostic tools
│   ├── debug_channels.py    # Channel detection and validation script
│   └── README.md            # Debug tools documentation
├── testing/                  # Testing and validation scripts
│   ├── test_consistency.sh  # Comprehensive pipeline consistency tests
│   └── quick_validate.sh    # Quick validation for core functionality
├── reports/                  # Generated reports and analysis
│   └── CONSISTENCY_REPORT.md # Detailed consistency check results
└── documentation/            # Documentation utilities
    └── read_pdf.py          # PDF content extraction utility
```

## Quick Reference

### Debug Tools (`debug/`)
- **`debug_channels.py`**: Validates input channel detection and image dimensions
  ```bash
  python tools/debug/debug_channels.py --dataset data/DFC2023Amini
  ```

### Testing Tools (`testing/`)
- **`test_consistency.sh`**: Comprehensive pipeline consistency validation
  ```bash
  ./tools/testing/test_consistency.sh
  ```
- **`quick_validate.sh`**: Quick validation of core functionality
  ```bash
  ./tools/testing/quick_validate.sh
  ```

### Reports (`reports/`)
- **`CONSISTENCY_REPORT.md`**: Detailed report of all consistency fixes and validations

### Documentation (`documentation/`)
- **`read_pdf.py`**: Utility for extracting text content from PDF documents

## Usage Guidelines

1. **Debug tools** should be used when troubleshooting pipeline issues
2. **Testing tools** should be run after making changes to ensure consistency
3. **Reports** contain historical analysis and should be referenced for understanding fixes
4. **Documentation tools** are utilities for project maintenance

## Git Tracking

- All Python scripts and documentation files are tracked in Git
- Output directories, logs, and temporary files are excluded via `.gitignore`
- See project root `.gitignore` for complete exclusion rules
