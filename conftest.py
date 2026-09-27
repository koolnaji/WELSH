# Makes the flat, package-less modules in this directory (language_detection,
# boilerplate, corpus_metadata, ...) importable from tests/ without needing
# an __init__.py or a src-layout install. pytest loads every conftest.py on
# the path to a collected test file before collection, so this runs before
# tests/test_*.py try to `import language_detection` etc.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
