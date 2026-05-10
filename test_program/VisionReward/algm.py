import json
from collections import defaultdict

# Read data
with open("dimension_result_20.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Group container
groups = defaultdict(list)

# Categorize (old / new / vpo / pav)
for item in data:
    path = item["video_path"]
    if "\\old\\" in path:
        groups["old"].append(item)
    elif "\\new\\" in path:
        groups["new"].append(item)
    elif "\\vpo\\" in path:
        groups["vpo"].append(item)
    elif "\\pav\\" in path:
        groups["pav"].append(item)

# Calculate averages
results = {}

for g in ["old", "new", "vpo", "pav"]:
    items = groups[g]
    n = len(items)

    sum_align = sum(x["Alignment"] for x in items)
    sum_stab = sum(x["Stability"] for x in items)
    sum_pres = sum(x["Preservation"] for x in items)
    sum_phys = sum(x["Physics"] for x in items)

    results[g] = {
        "count": n,
        "Alignment_avg": sum_align / n,
        "Stability_avg": sum_stab / n,
        "Preservation_avg": sum_pres / n,
        "Physics_avg": sum_phys / n,
        "formula": {
            "Alignment": f"{sum_align} / {n} = {sum_align / n:.4f}",
            "Stability": f"{sum_stab} / {n} = {sum_stab / n:.4f}",
            "Preservation": f"{sum_pres} / {n} = {sum_pres / n:.4f}",
            "Physics": f"{sum_phys} / {n} = {sum_phys / n:.4f}",
        }
    }

# Output results
import pprint
pprint.pprint(results)