# NISAR GCOV coverage audit (CMR) -> web/audit.json
import json, glob, os, sys
from shapely.geometry import Polygon, box
from shapely.validation import make_valid
S = sys.argv[1]
aoi = box(48.85, 38.85, 49.6, 39.6)
rows = {}
for f in sorted(glob.glob(os.path.join(S, "data", "cmr_*.json"))):
    coll = os.path.basename(f)[4:-5]
    level = "Beta" if "BETA" in coll else ("Provisional" if "PROVISIONAL" in coll else "Validated")
    for it in json.load(open(f))["items"]:
        g = it["umm"]; ur = g["GranuleUR"]; p = ur.split("_")
        pts = g["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"][0]["Boundary"]["Points"]
        poly = make_valid(Polygon([(q["Longitude"], q["Latitude"]) for q in pts]))
        att = {a["Name"]: a["Values"][0] for a in g.get("AdditionalAttributes", [])}
        key = f"{p[5]}-{p[6]}-{p[7]}-{p[8]}"
        d = f"{p[11][:4]}-{p[11][4:6]}-{p[11][6:8]}"
        r = rows.setdefault(key, dict(track=int(p[5]), dir="Ascending" if p[6] == "A" else "Descending", frame=int(p[7]),
                                      mode=p[8], pol=att.get("FREQUENCY_A_POLARIZATION_CONCAT"), bw=att.get("RANGE_BANDWIDTH_CONCAT"),
                                      acq={}, ))
        cov = round(poly.intersection(aoi).area / aoi.area, 2)
        prev = r["acq"].get(d)
        rank = {"Beta": 0, "Provisional": 1, "Validated": 2}
        if not prev or rank[level] > rank[prev["level"]]:
            r["acq"][d] = dict(level=level, cov=cov, granule=ur)
out = []
for k, r in sorted(rows.items()):
    acq = [dict(date=d, **v) for d, v in sorted(r["acq"].items())]
    r["acq"] = acq; r["id"] = k; r["n"] = len(acq); r["max_cov"] = max(a["cov"] for a in acq)
    out.append(r)
total = sum(r["n"] for r in out)
json.dump(dict(stacks=out, total_acq=total, queried="2026-11-14", collections=["NISAR_L2_GCOV_BETA_V1", "NISAR_L2_GCOV_PROVISIONAL_V1", "NISAR_L2_GCOV_V1"]),
          open(os.path.join(S, "web", "audit.json"), "w"))
for r in out: print(r["id"], r["n"], r["max_cov"], r["acq"][0]["date"], r["acq"][-1]["date"], r["pol"], r["bw"])
print("total", total)
