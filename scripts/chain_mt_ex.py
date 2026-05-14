from pathlib import Path 

inp = Path("~/ba_nylon/data/raw/d99r_f134w_d304m_r330a.pdb").expanduser()
out = Path("~/ba_nylon/inputs/nylc_chainB_mt.pdb").expanduser()
with open(inp) as f, open(out, "w") as g:
    for line in f:
        #selects chain
        if line.startswith(("ATOM", "HETATM")) and line[21] == "B" or line[21] == "C":
            g.write(line)
        elif line.startswith("TER"):
            continue
    g.write("TER\nEND\n")

