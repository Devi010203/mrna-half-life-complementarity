# Contributing

Please keep changes scoped and reproducible. New result claims should be traceable to fold-level predictions or metrics, and figure changes should preserve the associated plotted-data CSV output.

Before submitting a change:

1. Run `python scripts/check_setup.py`.
2. Run `python scripts/make_all_figures.py` for plotting changes.
3. Run Python bytecode compilation or targeted model smoke tests for code changes.
4. Document any new input file, environment variable, random seed, and output path.
5. Do not commit restricted datasets, model checkpoints, credentials, user-specific paths, or cluster account information.

