#!/usr/bin/env python3
"""
Command-line script to allow certain IDE-integrated code assist tools to run the FJC data processor.
This is a lightweight wrapper around the functionality in nomination_predictor.fjc_processor.

Please don't continue developing new functionality for the project's package in this file.
In virtually all cases new code would be better placed into the "real" file for which this is just a wrapper,
and in an ideal world I'd like to delete this whole file outright to avoid confusion.
In fact, if you ever determine there's no more need for this file, go ahead and delete it immediately.
If we ever need it back it'll likely become pretty obvious, and we can always retrieve it from git history later.
"""

from pathlib import Path
import sys

# Add the project root to the path if needed
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from nomination_predictor.fjc_processor import main

if __name__ == "__main__":
    main()
