# GitHub Repository Setup Guide

## Files to include (already configured in .gitignore)

### Essential code:
- experiment.py
- make_figures.py
- make_figure2_simple.py
- hyperparam_search.py

### Documentation:
- README.md
- LICENSE
- requirements.txt

### Figures (for reference):
- figure1_accuracy_curves.png
- figure2_signal_space.png

## Files excluded (by .gitignore):
- sweep_results*.pkl, .json, .csv (large data files - can be regenerated)
- hyperparam_search_results.json
- __pycache__/
- generate_figure2.py (obsolete script)
- figure2_signal_space_placeholder.png

## Setup Steps:

1. Initialize Git (from Experiment folder):
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Co-adaptive communication evolution"
   ```

2. Create GitHub repo:
   - Go to github.com
   - Create new repository (e.g., "coadaptive-communication")
   - Do NOT initialize with README (we have one)

3. Push to GitHub:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git branch -M main
   git push -u origin main
   ```

4. Update README.md with:
   - Your actual name
   - Your email/contact
   - Link to the paper (when published)
   - Citation info

## Repository size:
- ~1.5 MB (mostly the two PNG figures)
- Data files excluded (can be regenerated in ~10 min)

## Reproducibility:
Users can run `python experiment.py` to reproduce all results.
The .gitignore ensures only source code and documentation are tracked.
