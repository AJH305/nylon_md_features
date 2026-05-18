#!/usr/bin/env python3
"""
MD feature extraction for NylC catalytic competence filtering.

This script replaces the older generic/global features with a mechanistic MVP feature set:
local catalytic geometry, pocket accessibility proxies, F134/F134W mechanics,
termini electrostatics, interface stability proxies, local flexibility and hydration.

Assumptions:
- Trajectories are centered/aligned sufficiently for geometric interpretation.
- Residue numbering follows the NylCp2/NylC numbering used in the paper.
- Chain assignments can differ between systems. Adjust RESIDUE_MAP if needed.
- Pocket volume and SASA are implemented as lightweight proxies suitable for an MVP.
  For publication-grade pocket volume/SASA, validate against POVME/mdpocket/freesasa.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from MDAnalysis.lib.distances import calc_angles, calc_dihedrals, distance_array


# =========================
# Paths and settings
# =========================

PROJECT_ROOT = Path("/home/dwp46550/ba_nylon").resolve()
MD_DIR = PROJECT_ROOT / "data" / "md"
OUTPUT_DIR = PROJECT_ROOT / "results" / "processed_md_features_v1"

STRIDE = 10

# Distance thresholds in Angstrom.
HBOND_DISTANCE_CUTOFF = 3.5
SALT_BRIDGE_CUTOFF = 4.0
CONTACT_CUTOFF = 4.5
CATALYTIC_WATER_RADIUS = 5.0
POCKET_RADIUS = 8.0
POCKET_GRID_SPACING = 1.5
POCKET_PROTEIN_EXCLUSION = 2.0
GATE_OPEN_DISTANCE_CUTOFF = 10.0
PRECAT_THR_TO_SUBSTRATE_C_CUTOFF = 4.5
PRECAT_TYR_TO_SUBSTRATE_POLAR_CUTOFF = 4.0
PRECAT_LYS_TO_SUBSTRATE_POLAR_CUTOFF = 4.0

# Adjust this block if your chain assignment differs.
# Use chain=None to select all chains with a given resid. That is less specific and should be avoided if possible.
RESIDUE_MAP = {
    "D99": {"chain": "A", "resid": 99},
    "F134": {"chain": "A", "resid": 134},
    "Y146": {"chain": "A", "resid": 146},
    "K189": {"chain": "A", "resid": 189},
    "D191": {"chain": "A", "resid": 191},
    "T267": {"chain": "B", "resid": 267},
    "D304": {"chain": "B", "resid": 304},
    "D306": {"chain": "B", "resid": 306},
    "D308": {"chain": "B", "resid": 308},
    "R330": {"chain": "B", "resid": 330},
}

POCKET_RESIDUE_KEYS = ["D99", "F134", "Y146", "K189", "D191", "T267", "D304", "D306", "D308", "R330"]
GATE_RESIDUE_KEYS = ("F134", "Y146")
LOCAL_FLEXIBILITY_RESIDUE_KEYS = ["D99", "F134", "Y146", "K189", "D191", "T267", "D304", "D306", "D308", "R330"]
INTERFACE_RESIDUE_KEYS = ["D99", "F134", "Y146", "D191", "T267", "D304", "D306", "D308", "R330"]

ION_RESNAMES = "NA CL K MG CA ZN MN"
WATER_RESNAMES = "SOL WAT HOH TIP3 TIP3P"


# =========================
# Helper data structures
# =========================

@dataclass
class SelectionWarning:
    label: str
    selection: str
    message: str


# =========================
# Basic geometry helpers
# =========================

def safe_select(universe: mda.Universe, selection: str, label: str, required: bool = True):
    atoms = universe.select_atoms(selection)
    if len(atoms) == 0 and required:
        raise ValueError(f"Selection returned zero atoms for {label}: {selection}")
    return atoms


def select_chain_residue(
    universe: mda.Universe,
    label: str,
    chain: Optional[str],
    resid: int,
    atom_filter: Optional[str] = None,
    required: bool = True,
):
    """Select a residue robustly using chainID/segid first, then resid-only fallback."""
    atom_clause = f" and ({atom_filter})" if atom_filter else ""

    queries = []
    if chain is not None:
        queries.extend([
            f"protein and chainID {chain} and resid {resid}{atom_clause}",
            f"protein and segid {chain} and resid {resid}{atom_clause}",
        ])
    queries.append(f"protein and resid {resid}{atom_clause}")

    last_atoms = None
    for query in queries:
        atoms = universe.select_atoms(query)
        last_atoms = atoms
        if len(atoms) > 0:
            if query == queries[-1] and chain is not None:
                warnings.warn(
                    f"Used resid-only fallback for {label}. Check chain mapping. Query: {query}",
                    RuntimeWarning,
                )
            return atoms

    if required:
        raise ValueError(f"Could not select {label}. Tried: {queries}")
    return last_atoms


def select_residue_by_key(universe: mda.Universe, key: str, atom_filter: Optional[str] = None, required: bool = True):
    cfg = RESIDUE_MAP[key]
    return select_chain_residue(
        universe=universe,
        label=key,
        chain=cfg.get("chain"),
        resid=int(cfg["resid"]),
        atom_filter=atom_filter,
        required=required,
    )


def min_distance(group_a, group_b) -> float:
    if group_a is None or group_b is None or len(group_a) == 0 or len(group_b) == 0:
        return np.nan
    d = distance_array(group_a.positions, group_b.positions)
    return float(np.min(d))


def contact_count(group_a, group_b, cutoff: float = CONTACT_CUTOFF) -> int:
    if group_a is None or group_b is None or len(group_a) == 0 or len(group_b) == 0:
        return 0
    d = distance_array(group_a.positions, group_b.positions)
    return int(np.sum(d <= cutoff))


def has_contact(group_a, group_b, cutoff: float = CONTACT_CUTOFF) -> int:
    return int(contact_count(group_a, group_b, cutoff=cutoff) > 0)


def count_atoms_within(group_a, group_b, cutoff: float) -> int:
    if group_a is None or group_b is None or len(group_a) == 0 or len(group_b) == 0:
        return 0
    d = distance_array(group_a.positions, group_b.positions)
    return int(np.sum(np.any(d <= cutoff, axis=1)))


def centroid(atomgroup) -> np.ndarray:
    if atomgroup is None or len(atomgroup) == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return atomgroup.positions.astype(np.float64).mean(axis=0)


def distance_between_centroids(group_a, group_b) -> float:
    if group_a is None or group_b is None or len(group_a) == 0 or len(group_b) == 0:
        return np.nan
    return float(np.linalg.norm(centroid(group_a) - centroid(group_b)))


def angle_degrees(atom_a, atom_b, atom_c) -> float:
    if atom_a is None or atom_b is None or atom_c is None:
        return np.nan
    if len(atom_a) == 0 or len(atom_b) == 0 or len(atom_c) == 0:
        return np.nan
    angle = calc_angles(atom_a.positions[0], atom_b.positions[0], atom_c.positions[0])[0]
    return float(np.degrees(angle))


def dihedral_degrees(atom_a, atom_b, atom_c, atom_d):
    if len(atom_a) == 0 or len(atom_b) == 0 or len(atom_c) == 0 or len(atom_d) == 0:
        return np.nan

    dih = calc_dihedrals(
        atom_a.positions[0],
        atom_b.positions[0],
        atom_c.positions[0],
        atom_d.positions[0],
    )

    return float(np.degrees(np.asarray(dih).reshape(-1)[0]))


def classify_chi1(angle: float) -> str:
    if np.isnan(angle):
        return "missing"
    # Standard rough rotamer bins for chi1.
    if -90.0 <= angle < 30.0:
        return "gauche_minus"
    if 30.0 <= angle < 150.0:
        return "gauche_plus"
    return "trans"


def atom_names(atomgroup) -> set:
    if atomgroup is None or len(atomgroup) == 0:
        return set()
    return set(atomgroup.names)


# =========================
# Feature-specific helpers
# =========================

def get_sidechain_charged_atoms(residue_atoms):
    if residue_atoms is None or len(residue_atoms) == 0:
        return residue_atoms
    resnames = set(residue_atoms.resnames)
    if {"ARG"} & resnames:
        return residue_atoms.select_atoms("name NH1 NH2 NE CZ")
    if {"LYS"} & resnames:
        return residue_atoms.select_atoms("name NZ")
    if {"ASP"} & resnames:
        return residue_atoms.select_atoms("name OD1 OD2")
    if {"GLU"} & resnames:
        return residue_atoms.select_atoms("name OE1 OE2")
    # Fallback: use polar side-chain atoms only.
    return residue_atoms.select_atoms("not backbone and (name O* N*)")


def get_aromatic_sidechain_atoms(residue_atoms):
    if residue_atoms is None or len(residue_atoms) == 0:
        return residue_atoms
    return residue_atoms.select_atoms("not backbone and not name CB")


def nearest_atom_to_group(atomgroup, target_group):
    if atomgroup is None or target_group is None or len(atomgroup) == 0 or len(target_group) == 0:
        return None
    d = distance_array(atomgroup.positions, target_group.positions)
    idx = int(np.unravel_index(np.argmin(d), d.shape)[0])
    return atomgroup[idx:idx + 1]


def approximate_pocket_volume(protein_heavy, pocket_center: np.ndarray) -> float:
    """Voxel-based accessible-volume proxy around the active-site center.

    Counts grid points within POCKET_RADIUS of pocket_center that do not clash with nearby protein atoms.
    This is not a rigorous pocket-volume algorithm but is stable and dependency-light for an MVP.
    """
    if protein_heavy is None or len(protein_heavy) == 0 or np.any(np.isnan(pocket_center)):
        return np.nan

    r = POCKET_RADIUS
    spacing = POCKET_GRID_SPACING
    axis = np.arange(-r, r + spacing, spacing)
    grid = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    grid = grid[np.linalg.norm(grid, axis=1) <= r]
    points = grid + pocket_center

    nearby = protein_heavy.select_atoms(
        f"point {pocket_center[0]} {pocket_center[1]} {pocket_center[2]} {r + 4.0}"
    )
    if len(nearby) == 0:
        return float(len(points) * spacing ** 3)

    d = distance_array(points, nearby.positions)
    accessible = np.min(d, axis=1) >= POCKET_PROTEIN_EXCLUSION
    return float(np.sum(accessible) * spacing ** 3)


def sidechain_chi1(residue_atoms) -> Tuple[float, str]:
    if residue_atoms is None or len(residue_atoms) == 0:
        return np.nan, "missing"
    n = residue_atoms.select_atoms("name N")
    ca = residue_atoms.select_atoms("name CA")
    cb = residue_atoms.select_atoms("name CB")
    cg = residue_atoms.select_atoms("name CG")
    angle = dihedral_degrees(n, ca, cb, cg)
    return angle, classify_chi1(angle)


def local_residue_rmsd(residue_atoms, reference_positions: Dict[str, np.ndarray], key: str) -> float:
    ca = residue_atoms.select_atoms("name CA") if residue_atoms is not None and len(residue_atoms) > 0 else None
    if ca is None or len(ca) == 0 or key not in reference_positions:
        return np.nan
    return float(np.linalg.norm(ca.positions[0] - reference_positions[key]))


def compute_ca_rmsf(position_stack: np.ndarray) -> float:
    """RMSF for one atom over sampled frames."""
    if position_stack.ndim != 2 or position_stack.shape[0] < 2:
        return np.nan
    mean_pos = np.mean(position_stack, axis=0)
    sq = np.sum((position_stack - mean_pos) ** 2, axis=1)
    return float(np.sqrt(np.mean(sq)))


# =========================
# Selection logic
# =========================

def build_selections(u: mda.Universe, condition: str):
    protein = safe_select(u, "protein", "protein")
    backbone = safe_select(u, "protein and backbone", "backbone")
    protein_heavy = safe_select(u, "protein and not name H*", "protein_heavy")

    residue_atoms = {}
    for key in sorted(set(POCKET_RESIDUE_KEYS + LOCAL_FLEXIBILITY_RESIDUE_KEYS + INTERFACE_RESIDUE_KEYS)):
        residue_atoms[key] = select_residue_by_key(u, key, required=False)
        if len(residue_atoms[key]) == 0:
            warnings.warn(f"Residue {key} could not be selected. Its features will be NaN/0.", RuntimeWarning)

    atoms = {
        "T267_OG1": select_residue_by_key(u, "T267", "name OG1", required=False),
        "D306_carboxyl": select_residue_by_key(u, "D306", "name OD1 OD2 CG", required=False),
        "D306_OD": select_residue_by_key(u, "D306", "name OD1 OD2", required=False),
        "D308_carboxyl": select_residue_by_key(u, "D308", "name OD1 OD2 CG", required=False),
        "D308_OD": select_residue_by_key(u, "D308", "name OD1 OD2", required=False),
        "Y146_OH": select_residue_by_key(u, "Y146", "name OH", required=False),
        "K189_NZ": select_residue_by_key(u, "K189", "name NZ", required=False),
        "F134_ring": get_aromatic_sidechain_atoms(residue_atoms["F134"]),
        "F134W_NE1": residue_atoms["F134"].select_atoms("name NE1") if len(residue_atoms["F134"]) > 0 else residue_atoms["F134"],
        "D99_charged": get_sidechain_charged_atoms(residue_atoms["D99"]),
        "D191_charged": get_sidechain_charged_atoms(residue_atoms["D191"]),
        "D304_charged": get_sidechain_charged_atoms(residue_atoms["D304"]),
        "R330_charged": get_sidechain_charged_atoms(residue_atoms["R330"]),
    }

    triad_key_atoms = mda.AtomGroup([], u)
    for atom_key in ["T267_OG1", "D306_OD", "D308_OD"]:
        if len(atoms[atom_key]) > 0:
            triad_key_atoms += atoms[atom_key]

    pocket_residues = mda.AtomGroup([], u)
    for key in POCKET_RESIDUE_KEYS:
        if len(residue_atoms[key]) > 0:
            pocket_residues += residue_atoms[key]

    waters = u.select_atoms(f"resname {WATER_RESNAMES} and name O OW OH2")

    selections = {
        "protein": protein,
        "backbone": backbone,
        "protein_heavy": protein_heavy,
        "residue_atoms": residue_atoms,
        "atoms": atoms,
        "triad_key_atoms": triad_key_atoms,
        "pocket_residues": pocket_residues,
        "waters": waters,
    }

    if condition != "no_substrate":
        substrate = u.select_atoms(f"not protein and not resname {WATER_RESNAMES} {ION_RESNAMES}")
        if len(substrate) > 0:
            selections["substrate"] = substrate
            selections["substrate_heavy"] = substrate.select_atoms("not name H*")
            selections["substrate_oxygen"] = substrate.select_atoms("name O O* OT* OC* OXT")
            selections["substrate_nitrogen"] = substrate.select_atoms("name N N* NT*")
            selections["substrate_polar"] = substrate.select_atoms("name O O* N N* OT* OC* OXT NT*")
            selections["substrate_carbonyl_like_c"] = substrate.select_atoms("name C C* CA")

    return selections


# =========================
# Framewise feature calculation
# =========================

def calculate_frame_features(selections, reference_ca_positions: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
    protein_heavy = selections["protein_heavy"]
    residue_atoms = selections["residue_atoms"]
    atoms = selections["atoms"]
    triad_key_atoms = selections["triad_key_atoms"]
    pocket_residues = selections["pocket_residues"]
    waters = selections["waters"]

    substrate = selections.get("substrate")
    substrate_heavy = selections.get("substrate_heavy")
    substrate_oxygen = selections.get("substrate_oxygen")
    substrate_nitrogen = selections.get("substrate_nitrogen")
    substrate_polar = selections.get("substrate_polar")
    substrate_carbonyl_like_c = selections.get("substrate_carbonyl_like_c")

    t267 = atoms["T267_OG1"]
    d306_od = atoms["D306_OD"]
    d308_od = atoms["D308_OD"]
    y146_oh = atoms["Y146_OH"]
    k189_nz = atoms["K189_NZ"]
    f134_ring = atoms["F134_ring"]

    pocket_center = centroid(triad_key_atoms)
    if substrate_heavy is not None and len(substrate_heavy) > 0:
        # Bound-state center is biased toward substrate-proximal active site.
        pocket_center = np.nanmean(np.vstack([pocket_center, centroid(substrate_heavy)]), axis=0)

    gate_a = residue_atoms[GATE_RESIDUE_KEYS[0]].select_atoms("not backbone")
    gate_b = residue_atoms[GATE_RESIDUE_KEYS[1]].select_atoms("not backbone")
    gate_distance = distance_between_centroids(gate_a, gate_b)

    f134_chi1, f134_rotamer = sidechain_chi1(residue_atoms["F134"])

    features: Dict[str, float] = {
        # Local catalytic geometry.
        "cat_T267OG1_D306OD_min_dist_A": min_distance(t267, d306_od),
        "cat_T267OG1_D308OD_min_dist_A": min_distance(t267, d308_od),
        "cat_D306OD_D308OD_min_dist_A": min_distance(d306_od, d308_od),
        "cat_Y146OH_T267OG1_dist_A": min_distance(y146_oh, t267),
        "cat_K189NZ_T267OG1_dist_A": min_distance(k189_nz, t267),

        # Pocket accessibility proxies.
        "pocket_gate_F134_Y146_dist_A": gate_distance,
        "pocket_gate_open_state": int(not np.isnan(gate_distance) and gate_distance >= GATE_OPEN_DISTANCE_CUTOFF),
        "pocket_volume_voxel_proxy_A3": approximate_pocket_volume(protein_heavy, pocket_center),
        "active_site_water_accessibility_count_5A": count_atoms_within(waters, triad_key_atoms, CATALYTIC_WATER_RADIUS),
        "active_site_burial_atom_contacts_4p5A": contact_count(pocket_residues, protein_heavy, CONTACT_CUTOFF),

        # F134/F134W mechanics.
        "f134_chi1_deg": f134_chi1,
        "f134_rotamer_state": f134_rotamer,
        "f134_ring_to_T267OG1_centroid_dist_A": distance_between_centroids(f134_ring, t267),

        # Protein-internal electrostatics.
        "saltbridge_D99_D304_contact": has_contact(atoms["D99_charged"], atoms["D304_charged"], SALT_BRIDGE_CUTOFF),
        "saltbridge_D99_D304_min_dist_A": min_distance(atoms["D99_charged"], atoms["D304_charged"]),
        "saltbridge_D191_R330_contact": has_contact(atoms["D191_charged"], atoms["R330_charged"], SALT_BRIDGE_CUTOFF),
        "saltbridge_D191_R330_min_dist_A": min_distance(atoms["D191_charged"], atoms["R330_charged"]),

        # Interface stability proxy.
        "interface_central_residue_interchain_contacts_4p5A": count_interchain_contacts(residue_atoms, INTERFACE_RESIDUE_KEYS),
    }

    # Local flexibility as per-frame deviation from frame 0. True RMSF is computed after all frames.
    if reference_ca_positions is not None:
        for key in LOCAL_FLEXIBILITY_RESIDUE_KEYS:
            features[f"local_{key}_CA_displacement_from_ref_A"] = local_residue_rmsd(
                residue_atoms[key], reference_ca_positions, key
            )

    if substrate is not None and len(substrate) > 0:
        nearest_substrate_c_to_t267 = nearest_atom_to_group(substrate_carbonyl_like_c, t267)
        nearest_substrate_o_to_t267 = nearest_atom_to_group(substrate_oxygen, t267)
        nearest_substrate_polar_to_y146 = nearest_atom_to_group(substrate_polar, y146_oh)

        features.update({
            # Local catalytic geometry to substrate analog.
            "cat_T267OG1_substrateC_min_dist_A": min_distance(t267, substrate_carbonyl_like_c),
            "cat_T267OG1_substrateO_min_dist_A": min_distance(t267, substrate_oxygen),
            "cat_Y146OH_substrate_polar_min_dist_A": min_distance(y146_oh, substrate_polar),
            "cat_K189NZ_substrate_polar_min_dist_A": min_distance(k189_nz, substrate_polar),
            "cat_T267OG1_substrateC_substrateO_angle_deg": angle_degrees(
                t267, nearest_substrate_c_to_t267, nearest_substrate_o_to_t267
            ),
            "cat_Y146OH_substrate_polar_T267OG1_angle_deg": angle_degrees(
                y146_oh, nearest_substrate_polar_to_y146, t267
            ),

            # Pre-catalytic state occupancy flag per frame.
            "precatalytic_state_flag": int(
                min_distance(t267, substrate_carbonyl_like_c) <= PRECAT_THR_TO_SUBSTRATE_C_CUTOFF
                and min_distance(y146_oh, substrate_polar) <= PRECAT_TYR_TO_SUBSTRATE_POLAR_CUTOFF
                and min_distance(k189_nz, substrate_polar) <= PRECAT_LYS_TO_SUBSTRATE_POLAR_CUTOFF
            ),

            # Substrate accessibility and binding contacts.
            "substrate_to_pocket_contacts_4p5A": contact_count(substrate_heavy, pocket_residues, CONTACT_CUTOFF),
            "substrate_to_triad_min_dist_A": min_distance(substrate_heavy, triad_key_atoms),

            # F134/F134W mechanics to amide/polar substrate atoms.
            "f134_ring_to_substrate_polar_min_dist_A": min_distance(f134_ring, substrate_polar),
            "f134W_NE1_substrateO_hbond_proxy": has_contact(atoms["F134W_NE1"], substrate_oxygen, HBOND_DISTANCE_CUTOFF),
            "f134_fixing_state_flag": int(
                min_distance(f134_ring, substrate_polar) <= 4.5
                or has_contact(atoms["F134W_NE1"], substrate_oxygen, HBOND_DISTANCE_CUTOFF)
            ),
            "f134_open_space_state_flag": int(
                not np.isnan(gate_distance)
                and gate_distance >= GATE_OPEN_DISTANCE_CUTOFF
                and min_distance(f134_ring, substrate_polar) > 4.5
            ),

            # Termini electrostatics. These are contact proxies because exact terminal atom naming differs.
            "D99_substrate_polar_contact": has_contact(atoms["D99_charged"], substrate_polar, SALT_BRIDGE_CUTOFF),
            "D99_substrate_polar_min_dist_A": min_distance(atoms["D99_charged"], substrate_polar),
            "R330_substrate_polar_contact": has_contact(atoms["R330_charged"], substrate_polar, SALT_BRIDGE_CUTOFF),
            "R330_substrate_polar_min_dist_A": min_distance(atoms["R330_charged"], substrate_polar),
            "D99_substrate_N_contact": has_contact(atoms["D99_charged"], substrate_nitrogen, SALT_BRIDGE_CUTOFF),
            "R330_substrate_O_contact": has_contact(atoms["R330_charged"], substrate_oxygen, SALT_BRIDGE_CUTOFF),

            # Hydration features near catalytic atoms and substrate attack-site proxy.
            "water_count_within_3p5A_triad": count_atoms_within(waters, triad_key_atoms, 3.5),
            "water_count_within_5A_triad": count_atoms_within(waters, triad_key_atoms, 5.0),
            "water_count_within_5A_substrate_near_T267": count_atoms_within(waters, nearest_substrate_c_to_t267, 5.0),
            "catalytic_water_present_3p5A": int(count_atoms_within(waters, triad_key_atoms, 3.5) > 0),
        })
    else:
        # Apo systems still get hydration around the catalytic triad.
        features.update({
            "water_count_within_3p5A_triad": count_atoms_within(waters, triad_key_atoms, 3.5),
            "water_count_within_5A_triad": count_atoms_within(waters, triad_key_atoms, 5.0),
            "catalytic_water_present_3p5A": int(count_atoms_within(waters, triad_key_atoms, 3.5) > 0),
        })

    return features


def count_interchain_contacts(residue_atoms: Dict[str, object], keys: Iterable[str], cutoff: float = CONTACT_CUTOFF) -> int:
    """Count heavy-atom contacts between configured central residues from different chains/segids."""
    groups = []
    for key in keys:
        atoms = residue_atoms.get(key)
        if atoms is None or len(atoms) == 0:
            continue
        heavy = atoms.select_atoms("not name H*")
        if len(heavy) == 0:
            continue
        chain_ids = set(getattr(heavy, "chainIDs", []))
        segids = set(getattr(heavy, "segids", []))
        chain_label = next(iter(chain_ids - {""}), None) or next(iter(segids - {""}), None) or "unknown"
        groups.append((key, chain_label, heavy))

    total = 0
    for i, (_, chain_i, group_i) in enumerate(groups):
        for _, chain_j, group_j in groups[i + 1:]:
            if chain_i != chain_j:
                total += contact_count(group_i, group_j, cutoff=cutoff)
    return int(total)


# =========================
# Aggregation logic
# =========================

def summarize_frame_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate framewise features to one row per replicate.

    Occupancy/contact/state flags are summarized by mean, which equals fraction of sampled frames.
    Continuous features are summarized by mean, std, median, q05 and q95.
    """
    metadata_cols = ["variant", "condition", "state", "replicate"]
    numeric_cols = [c for c in df.columns if c not in metadata_cols + ["frame", "time_ps"] and pd.api.types.is_numeric_dtype(df[c])]

    summary = {col: df[col].iloc[0] for col in metadata_cols if col in df.columns}
    summary["n_sampled_frames"] = int(len(df))

    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) == 0:
            summary[f"{col}_mean"] = np.nan
            summary[f"{col}_std"] = np.nan
            summary[f"{col}_median"] = np.nan
            summary[f"{col}_q05"] = np.nan
            summary[f"{col}_q95"] = np.nan
            continue
        summary[f"{col}_mean"] = float(values.mean())
        summary[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary[f"{col}_median"] = float(values.median())
        summary[f"{col}_q05"] = float(values.quantile(0.05))
        summary[f"{col}_q95"] = float(values.quantile(0.95))

    # State ratios with clearer names.
    rename_mean_to_occupancy = [
        "precatalytic_state_flag",
        "pocket_gate_open_state",
        "f134_fixing_state_flag",
        "f134_open_space_state_flag",
        "f134W_NE1_substrateO_hbond_proxy",
        "D99_substrate_polar_contact",
        "R330_substrate_polar_contact",
        "D99_substrate_N_contact",
        "R330_substrate_O_contact",
        "saltbridge_D99_D304_contact",
        "saltbridge_D191_R330_contact",
        "catalytic_water_present_3p5A",
    ]
    for base in rename_mean_to_occupancy:
        old = f"{base}_mean"
        if old in summary:
            summary[f"{base}_occupancy"] = summary[old]

    return pd.DataFrame([summary])


def summarize_rotamers(df: pd.DataFrame) -> pd.DataFrame:
    if "f134_rotamer_state" not in df.columns or len(df) == 0:
        return pd.DataFrame()
    meta = {col: df[col].iloc[0] for col in ["variant", "condition", "state", "replicate"] if col in df.columns}
    counts = df["f134_rotamer_state"].value_counts(normalize=True, dropna=False).to_dict()
    row = dict(meta)
    for state in ["gauche_minus", "gauche_plus", "trans", "missing"]:
        row[f"f134_rotamer_{state}_occupancy"] = float(counts.get(state, 0.0))
    return pd.DataFrame([row])


def compute_local_rmsf_table(ca_positions: Dict[str, List[np.ndarray]], metadata: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for key, positions in ca_positions.items():
        if len(positions) == 0:
            rmsf_value = np.nan
        else:
            rmsf_value = compute_ca_rmsf(np.vstack(positions))
        cfg = RESIDUE_MAP[key]
        rows.append({
            **metadata,
            "residue_key": key,
            "chain_config": cfg.get("chain"),
            "resid": cfg.get("resid"),
            "ca_rmsf_A": rmsf_value,
            "n_sampled_frames": len(positions),
        })
    return pd.DataFrame(rows)


# =========================
# Processing
# =========================

def process_replicate(variant: str, condition: str, replicate: int, pdb_file: Path, xtc_file: Path):
    print("\n==============================")
    print(f"Variant: {variant}")
    print(f"Condition: {condition}")
    print(f"Replicate: {replicate}")
    print(f"PDB: {pdb_file}")
    print(f"XTC: {xtc_file}")

    u = mda.Universe(str(pdb_file), str(xtc_file))
    print(f"Atoms: {u.atoms.n_atoms}")
    print(f"Frames: {len(u.trajectory)}")

    selections = build_selections(u, condition)

    # Reference CA positions for local displacement proxies.
    u.trajectory[0]
    reference_ca_positions = {}
    for key in LOCAL_FLEXIBILITY_RESIDUE_KEYS:
        ca = selections["residue_atoms"][key].select_atoms("name CA") if len(selections["residue_atoms"][key]) > 0 else None
        if ca is not None and len(ca) > 0:
            reference_ca_positions[key] = ca.positions[0].copy()

    rows = []
    ca_positions = {key: [] for key in LOCAL_FLEXIBILITY_RESIDUE_KEYS}

    for sampled_frame_idx, ts in enumerate(u.trajectory[::STRIDE]):
        features = calculate_frame_features(
            selections,
            reference_ca_positions=reference_ca_positions,
        )

        features.update({
            "variant": variant,
            "condition": condition,
            "state": "apo" if condition == "no_substrate" else "bound",
            "replicate": replicate,
            "frame": sampled_frame_idx,
            "trajectory_frame": int(ts.frame),
            "time_ps": float(ts.time),
        })
        rows.append(features)

        for key in LOCAL_FLEXIBILITY_RESIDUE_KEYS:
            atoms = selections["residue_atoms"][key]
            ca = atoms.select_atoms("name CA") if len(atoms) > 0 else None
            if ca is not None and len(ca) > 0:
                ca_positions[key].append(ca.positions[0].copy())

    frame_df = pd.DataFrame(rows)
    summary_df = summarize_frame_features(frame_df)
    rotamer_df = summarize_rotamers(frame_df)
    rmsf_df = compute_local_rmsf_table(
        ca_positions,
        metadata={
            "variant": variant,
            "condition": condition,
            "state": "apo" if condition == "no_substrate" else "bound",
            "replicate": replicate,
        },
    )

    out_dir = OUTPUT_DIR / variant / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_file = out_dir / f"{variant}_{condition}_rep{replicate}_frame_features_v1.csv"
    summary_file = out_dir / f"{variant}_{condition}_rep{replicate}_summary_features_v1.csv"
    rotamer_file = out_dir / f"{variant}_{condition}_rep{replicate}_f134_rotamers_v1.csv"
    rmsf_file = out_dir / f"{variant}_{condition}_rep{replicate}_local_rmsf_v1.csv"

    frame_df.to_csv(frame_file, index=False)
    summary_df.to_csv(summary_file, index=False)
    rotamer_df.to_csv(rotamer_file, index=False)
    rmsf_df.to_csv(rmsf_file, index=False)

    print(f"Saved frame features: {frame_file}")
    print(f"Saved summary features: {summary_file}")
    print(f"Saved F134 rotamers: {rotamer_file}")
    print(f"Saved local RMSF: {rmsf_file}")
    print("==============================")


def main():
    print(f"Using MD directory: {MD_DIR}")
    print(f"Using output directory: {OUTPUT_DIR}")

    if not MD_DIR.exists():
        raise FileNotFoundError(f"MD directory does not exist: {MD_DIR}")

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

                process_replicate(
                    variant=variant,
                    condition=condition,
                    replicate=replicate,
                    pdb_file=pdb_file,
                    xtc_file=xtc_file,
                )


if __name__ == "__main__":
    main()
