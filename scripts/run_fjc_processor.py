#!/usr/bin/env python3
"""
Command-line script to run the FJC data processor.

This is a lightweight wrapper around the functionality in nomination_predictor.fjc_processor.
"""

import sys
from pathlib import Path

# Add the project root to the path if needed
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from nomination_predictor.fjc_processor import main

if __name__ == "__main__":
    main()
