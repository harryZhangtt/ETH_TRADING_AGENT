"""
rewards/tests — test suite for the rewards package.

Run all tests:
    python rewards/tests/__init__.py          # direct
    pytest rewards/tests/ -v                  # via pytest from project root
"""

import os
import sys

if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([os.path.dirname(__file__), "-v"]))
