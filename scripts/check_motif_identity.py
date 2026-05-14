from pathlib import Path
import re

pdb_dir = Path("~/ba_nylon/proteindj_runs/results/nylc_mt_motifscaff_test/unpacked_ranked/ranked_designs").expanduser()

expected = {
    1: "THR",    # original B267
    43: "ASP",  # expected original B306, approximate for this contig
    45: "ASP",  # expected original B308, approximate for this contig
}

def get_residues(pdb_path):
    residues = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                chain = line[21].strip()
                resname = line[17:20].strip()
                resid = int(line[22:26].strip())
                residues[(chain, resid)] = resname
    return residues

print("file,status,A1,A43,A45")

for pdb in sorted(pdb_dir.glob("*.pdb")):
    residues = get_residues(pdb)

    observed = {}
    ok = True

    for resid, exp_resname in expected.items():
        obs = residues.get(("A", resid), "MISSING")
        observed[resid] = obs
        if obs != exp_resname:
            ok = False

    status = "OK" if ok else "FAIL"

    print(
        f"{pdb.name},{status},"
        f"{observed[1]},"
        f"{observed[43]},"
        f"{observed[45]}"
    )
