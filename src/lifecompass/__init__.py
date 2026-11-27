import sys

# No __pycache__ is written by any entry point of the system (the listener, the scheduled tasks, the scripts): the bytecode is
# compiled in memory at each start, which costs milliseconds, and no generated files appear in the project (design decision 2026-11-19).
sys.dont_write_bytecode = True
