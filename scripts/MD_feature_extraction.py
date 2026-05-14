#!/usr/bin/env python3

from pathlib import Path
import re
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from MDAnalysis.analysis import rms


# =========================
# Paths and settings
# =========================

PROJECT_ROOT = Path("/home/dwp46550/ba_nylon").resolve()
MD_DIR = PROJECT_ROOT / "data" / "md"
OUTPUT_DIR = PROJECT_ROOT / "results" / "processed_md_features"

STRIDE = 10


# =========================
# Helper functions // can be changed according to needs
# =========================

#checks if Atom selection is empty
def safe_select(universe, selection, label):
    atoms = universe.select_atoms(selection)
    if len(atoms) == 0:
        raise ValueError(f"Selection returned zero atoms for {label}: {selection}")
    return atoms

#ensures that Atomgroup is returnes as float64 datatype
def get_positions(atomgroup):
    return atomgroup.positions.astype(np.float64)

# calculates minimal distance between atom groups
def min_distance(group_a, group_b):
    d = distance_array(group_a.positions, group_b.positions)
    return float(np.min(d))

#counts every atom-atom-pairs in between two groups, which are closer together than the cutooff value
def contact_count(group_a, group_b, cutoff=4.0):
    d = distance_array(group_a.positions, group_b.positions)
    return int(np.sum(d <= cutoff))


def count_atoms_within(group_a, group_b, cutoff):
    d = distance_array(group_a.positions, group_b.positions)
    return int(np.sum(np.any(d <= cutoff, axis=1)))


def triangle_area(p1, p2, p3):
    return float(0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1)))


def get_triad_geometry(triad_atoms):
    positions = get_positions(triad_atoms)

    if len(positions) < 3:
        return {
            "triad_d01": np.nan,
            "triad_d02": np.nan,
            "triad_d12": np.nan,
            "triad_area": np.nan,
        }

    p0, p1, p2 = positions[0], positions[1], positions[2]

    return {
        "triad_d01": float(np.linalg.norm(p0 - p1)),
        "triad_d02": float(np.linalg.norm(p0 - p2)),
        "triad_d12": float(np.linalg.norm(p1 - p2)),
        "triad_area": triangle_area(p0, p1, p2),
    }


def hbond_proxy_count(group_a, group_b, cutoff=3.5):
    d = distance_array(group_a.positions, group_b.positions)
    return int(np.sum(d <= cutoff))


# =========================
# Feature logic 
# =========================

#defines ones for each trajectory which aromgroups are analysed
def build_selections(u, condition):
    protein = safe_select(u, "protein", "protein")
    backbone = safe_select(u, "protein and backbone", "backbone")

    #needs to be changed 
    triad_atoms = safe_select(
    u,
    "protein and ("
    "(resid 267 and name OG1) or "
    "(resid 306 and name CG) or "
    "(resid 308 and name CG)"
    ")",
    "triad_atoms",
    )

    local_env = safe_select(
        u,
        "protein and around 8.0 "
        "(protein and ((resid 267) or (resid 306) or (resid 308)))",
        "local_env",
    )

    selections = {
        "protein": protein,
        "backbone": backbone,
        "triad_atoms": triad_atoms,
        "local_env": local_env,
    }

    waters = u.select_atoms("resname SOL WAT HOH and name O OW")
    if len(waters) > 0:
        selections["waters"] = waters

    if condition != "no_substrate":
        substrate = u.select_atoms(
            "not protein and not resname SOL WAT HOH NA CL K"
        )

        if len(substrate) > 0:
            selections["substrate"] = substrate
            selections["substrate_polar"] = substrate.select_atoms("name O N")

    return selections

#calculates features per frame
def calculate_frame_features(selections, reference_backbone_positions=None):
    protein = selections["protein"]
    backbone = selections["backbone"]
    triad_atoms = selections["triad_atoms"]
    local_env = selections["local_env"]

    features = {
        "protein_rg": float(protein.radius_of_gyration()),
        "local_env_rg": float(local_env.radius_of_gyration()),
        "local_env_n_atoms": int(len(local_env)),
        "local_env_to_triad_min_distance": min_distance(local_env, triad_atoms),
    }

    if reference_backbone_positions is not None:
        features["backbone_rmsd"] = float(
            rms.rmsd(
                backbone.positions,
                reference_backbone_positions,
                center=True,
                superposition=True,
            )
        )

    features.update(get_triad_geometry(triad_atoms))

    if "waters" in selections:
        waters = selections["waters"]
        features["water_oxygen_count_within_5A_triad"] = count_atoms_within(
            waters, triad_atoms, cutoff=5.0
        )
        features["water_oxygen_count_within_8A_triad"] = count_atoms_within(
            waters, triad_atoms, cutoff=8.0
        )

    if "substrate" in selections:
        substrate = selections["substrate"]

        features.update({
            "substrate_rg": float(substrate.radius_of_gyration()),
            "substrate_to_triad_min_distance": min_distance(substrate, triad_atoms),
            "substrate_to_local_env_contacts_4A": contact_count(
                substrate, local_env, cutoff=4.0
            ),
            "substrate_to_local_env_contacts_5A": contact_count(
                substrate, local_env, cutoff=5.0
            ),
        })

        substrate_polar = selections.get("substrate_polar")
        if substrate_polar is not None and len(substrate_polar) > 1:
            features["substrate_intramolecular_hbond_proxy"] = hbond_proxy_count(
                substrate_polar,
                substrate_polar,
                cutoff=3.5,
            )

    return features


# =========================
# Processing
# =========================

def process_replicate(variant, condition, replicate, pdb_file, xtc_file):
    print("\n==============================")
    print(f"Variant: {variant}")
    print(f"Condition: {condition}")
    print(f"Replicate: {replicate}")
    print(f"PDB: {pdb_file}")
    print(f"XTC: {xtc_file}")
    #universe Object used for efficient computing otherwise RAM usage would be off the charts
    u = mda.Universe(str(pdb_file), str(xtc_file))

    print(f"Atoms: {u.atoms.n_atoms}")
    print(f"Frames: {len(u.trajectory)}")

    selections = build_selections(u, condition)

    u.trajectory[0]
    reference_backbone_positions = selections["backbone"].positions.copy()

    rows = []
    #framewise feature calculation
    for frame_idx, ts in enumerate(u.trajectory[::STRIDE]):
        features = calculate_frame_features(
            selections,
            reference_backbone_positions=reference_backbone_positions,
        )

        features.update({
            "variant": variant,
            "condition": condition,
            "state": "apo" if condition == "no_substrate" else "bound",
            "replicate": replicate,
            "frame": frame_idx,
            "time_ps": float(ts.time),
        })

        rows.append(features)
    #saves features as .csv file for further analysis
    df = pd.DataFrame(rows)

    out_dir = OUTPUT_DIR / variant / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{variant}_{condition}_rep{replicate}_features.csv"
    df.to_csv(out_file, index=False)

    print(f"Saved: {out_file}")
    print("==============================")


def main():
    print(f"Using MD directory: {MD_DIR}")
    print(f"Using output directory: {OUTPUT_DIR}")
    #change loop structure accoording too your directory structure
    for variant_dir in sorted(MD_DIR.iterdir()):
        if not variant_dir.is_dir():
            continue

        variant = variant_dir.name

        for condition_dir in sorted(variant_dir.iterdir()):
            if not condition_dir.is_dir():
                continue

            condition = condition_dir.name

            for xtc_file in sorted(condition_dir.glob("md_center_*.xtc")):
                match = re.search(r"md_center_(\d+)", xtc_file.stem)

                if match is None:
                    print(f"Could not detect replicate number from {xtc_file.name}")
                    continue

                replicate = int(match.group(1))
                pdb_file = condition_dir / f"md_center_{replicate}.pdb"

                if not pdb_file.exists():
                    print(f"Missing matching PDB for {xtc_file.name}")
                    continue
                    #process function
                process_replicate(
                    variant=variant,
                    condition=condition,
                    replicate=replicate,
                    pdb_file=pdb_file,
                    xtc_file=xtc_file,
                )


if __name__ == "__main__":
    main()