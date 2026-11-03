"""Check that the notebooks agree on where the project lives on Drive.

Every notebook must contain the same project-folder block (DRIVE_ROOT) and, where it uses the
experiment layout, the same layout block (ERM_ID / ERM_DIR / first_existing). Run from the repository root:  python tools/check_notebooks.py
"""
import json
import pathlib
import sys

ROOT_MARK = "DRIVE_ROOT_NAMES = "
LAYOUT_START = "# Drive layout (Nov 2026): waterbirds/<ERM_ID>/"


def block(src, start, end_marker):
    i = src.find(start)
    if i < 0:
        return None
    j = src.find(end_marker, i)
    return src[i:j if j > 0 else None]


def main():
    nbs = sorted(pathlib.Path("notebooks").rglob("*.ipynb"))
    roots, layouts, problems = {}, {}, []
    for p in nbs:
        src = "\n".join("".join(c["source"]) for c in json.load(open(p, encoding="utf-8"))["cells"] if c["cell_type"] == "code")
        r = block(src, ROOT_MARK, 'print("Project folder on Drive:"')
        if r is None:
            problems.append(f"{p}: no shared project-folder block (DRIVE_ROOT_NAMES)")
        else:
            roots[p] = r
        l = block(src, LAYOUT_START, "return Path(paths[0])")
        if l is not None:
            layouts[p] = l
    for name, blocks in (("project-folder", roots), ("layout", layouts)):
        if len(set(blocks.values())) > 1:
            problems.append(f"the {name} block differs between: " + ", ".join(str(k) for k in blocks))
    for line in problems:
        print("✗", line)
    print(f"{len(nbs)} notebooks, {len(roots)} with the project-folder block, {len(layouts)} with the layout block.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
