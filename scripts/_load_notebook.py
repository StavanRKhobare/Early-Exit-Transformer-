# Load the study's functions from experiment.ipynb, stopping before its Run section.
import json, os

def notebook_source(path="experiment.ipynb"):
    out = []
    for c in json.load(open(path))["cells"]:
        text = "".join(c["source"])
        if c["cell_type"] == "markdown" and "## 11. Run" in text:
            break
        if c["cell_type"] == "code":
            out.append("\n".join(l for l in text.split("\n") if not l.lstrip().startswith("!")))
    return "\n".join(out)
