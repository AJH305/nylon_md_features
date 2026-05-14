from pathlib import Path 

inp = Path("~/ba_nylon/data/raw/wt.pdb").expanduser()
out = Path("~/ba_nylon/inputs/nylc_chainB_wt.pdb").expanduser()

with open(inp) as f, open(out, "w") as g:
    for line in f:
        #selects chain
        if line.startswith(("ATOM", "HETATM")) and line[21] == "B":
            g.write(line)
        elif line.startswith("TER"):
            continue
    g.write("TER\nEND\n")

