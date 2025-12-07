Thank you for contributing to CR-CA!

Please include the following in your pull request:

- Description: Short summary of the change and motivation.
- Issue: Link to the issue number it fixes (if applicable).
- Testing: How the change was tested (unit tests, sanity_check, manual).
- Dependencies: Any new or updated dependencies.
- CI: Ensure `CR-CA/sanity_check.py` and `CR-CA/test_core.py` pass locally.

Before submitting:
- Run `python3 sanity_check.py` and `pytest -q test_core.py`.
- Run `make format` / `black` if available.

Maintainers:
- For CRCA engine changes: @IlumCI

Notes:
- Keep PRs small and focused. For proposals that change defaults or public APIs, include rationale and a short migration note.
Thank you for contributing to CRCA!
