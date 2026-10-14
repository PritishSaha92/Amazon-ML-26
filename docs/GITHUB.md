# Uploading to GitHub

The repository is prepared for a source-code upload. The local submission TSVs remain in `output/` and are ignored, along with datasets, model weights, caches, and runtime logs. The two shipped model metadata JSON files and documentation figures are included.

## First upload

Create an empty repository on GitHub, then run these commands from this project's root. Replace the remote URL with your repository URL:

```bash
git init -b main
git add .
git status --short
git diff --cached --stat
```

Review the staged files, then commit and push:

```bash
git commit -m "Organize business entity resolution solution"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

For an existing Git repository, use its current branch and remote configuration. Ignore rules do not remove files already tracked by Git.

## Layout changes from the submission archive

| Original location | Repository location |
| --- | --- |
| `code/business_entity_resolution/src/src/` | `src/` |
| `code/business_entity_resolution/src/logs/*.sh` | `scripts/` |
| Other runtime folders under `code/business_entity_resolution/src/` | Same folder names at repository root |
| `code/business_entity_resolution/requirements.txt` | `requirements.txt` |
| `code/business_entity_resolution/README.md` | `docs/REPRODUCTION.md` |
| `Documentation_template.md` | `docs/SOLUTION.md` |
| `doc_figures/` | `docs/figures/` |
| `Public Leaderboard Final/img.jpeg` | `docs/figures/public-leaderboard.jpeg` |

Set `BER_ROOT` to the repository root. Script references, figure links, and the France-builder checksum guard have been updated for this layout. `logs/` is now reserved for runtime logs. Run launchers with Bash in Linux or WSL2.

The project licence remains unspecified; the team can add a licence when it has chosen the intended reuse terms.
