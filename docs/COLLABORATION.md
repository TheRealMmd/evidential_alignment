# Working together: repository, notebooks and Drive

Three people run experiments from Colab on one shared Drive folder. This is how we keep the code, the
notebooks and the outputs in step.

## 1. One home for each kind of thing
| What | Lives in | Never in |
|---|---|---|
| Code and notebooks | this GitHub repository | Drive, personal Colab copies |
| Data and run outputs | the shared Drive folder (`FeatureRater/`, see DRIVE_LAYOUT.md) | the repository |
| The plan and who runs what | the FeatureRater sheet (Rater / Classifier tabs) | chat threads |

## 2. Opening a notebook
Open it from GitHub, not from a copy in your Drive:
`https://colab.research.google.com/github/TheRealMmd/evidential_alignment/blob/<branch>/notebooks/<folder>/<name>.ipynb`
(Colab: File › Open notebook › GitHub). The notebook clones the same branch, so code and notebook match.
Each run records the commit it used in its outputs.

First time only: in Drive, open the shared folder and choose *Add shortcut to Drive* → My Drive. Name the
shortcut `FeatureRater` (the old name also works).

## 3. Two kinds of change
**Settings** (values in the configuration cells: seeds, outer loss, selection rule, learning rate …):
change them in Colab and run. Do not save the notebook back. The run's `config.json` records every setting,
so nothing is lost and nobody else is affected.

**Structural changes** (code in `algorithms/` or `analysis/`, new or removed notebook cells, paths, the Drive
layout, anything another person's run depends on):
1. Make a branch (`<name>/<topic>`), change and test.
2. Open a pull request; one of the other two reviews it.
3. Merge to `main`, then tell the others to reopen their notebooks from `main`.
4. Drive layout changes happen only after the code that understands them is merged, and only as moves
   (never copies or deletes): the Drive has no spare space.

`python tools/check_notebooks.py` checks that every notebook uses the same project-folder and layout code.

## 4. Claiming and recording runs
Before you start a run, set its row in the sheet to `Running` with your name in *Owner*. When it finishes,
fill in the Drive folder and the numbers, and set `Done`. Two people never run the same row.

## 5. Old notebooks
- Personal copies in "Colab Notebooks" (each person's own Drive): rename them with an `OLD ` prefix or move
  them into an `old notebooks` folder, and stop using them. They point to folders that moved on 3 Nov.
- The notebooks removed from the repository in the cleanup stay in git history (`git log -- old_notebooks`).
- Until the cleanup branch is merged, `main` still has notebooks with the old Drive paths; use the branch.
