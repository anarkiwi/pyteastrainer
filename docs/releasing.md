# Releasing

Publishing to PyPI uses [trusted publishing](https://docs.pypi.org/trusted-publishers/)
from `.github/workflows/release.yml`; no API token is stored in the repository.

## One-time setup

1. On PyPI, add a trusted publisher for project `pyteastrainer`
   (a *pending* publisher before the first upload):
   owner `anarkiwi`, repository `pyteastrainer`, workflow `release.yml`,
   environment `pypi`.
2. In the GitHub repository settings, create the environment `pypi`
   (optionally with required reviewers).

## Each release

1. Bump `version` in `pyproject.toml` and merge to `main`.
2. Create a GitHub release with tag `v<version>`; publishing the release
   builds the sdist and wheel and uploads them to PyPI.
