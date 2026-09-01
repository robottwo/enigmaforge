#!/usr/bin/env python3
"""Command-line entry point for EnigmaForge.

    ./run.py --size small --seed 2026 --mode story --out runs/demo-story

Equivalent to `python3 -m enigmaforge.pipeline` (which stays available);
importing the module directly here avoids runpy's double-execution path.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enigmaforge.pipeline import main

if __name__ == "__main__":
    main()
