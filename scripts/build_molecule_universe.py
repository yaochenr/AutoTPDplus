"""Build docs/assets/molecules.json, the website background, from experimental PROTAC structures.

Each SDF in data/protac_reference is a ligand from a PDB entry, named <pdb>_<chain>_<ligand>.sdf.
Files that share a PDB ID are conformers of one molecule, so one is kept per entry. Heavy-atom
coordinates are the deposited ones. Most depositions carry no hydrogens, so missing hydrogens are
added and only those new atoms are relaxed.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
SDF_DIR = ROOT / "data" / "protac_reference"
OUT = ROOT / "docs" / "assets" / "molecules.json"

ANGSTROM_PER_UNIT = 1.5
DOUBLE_OFFSET = 0.16
QUANT = 50
CLASH_VDW_FRACTION = 0.7

ELEMENT_CLASS = {"C": 0, "N": 1, "O": 2, "S": 3, "P": 3, "F": 4, "Cl": 4, "Br": 4, "I": 4, "H": 6}

# Ligands only partly modelled in the deposited structure (heavy atoms present / in the RCSB
# chemical component), checked 2026-11-04. Adding hydrogens would cap the break as if it were
# real chemistry. New SDFs need the same check before they are added.
INCOMPLETE = {
    "6wwb_E_YA3": (37, 61),
    "8dsf_J_TO0": (38, 81),
    "8dsf_L_TO0": (38, 81),
    "8ood_J_VY3": (43, 58),
    "8ood_K_VY3": (32, 58),
    "8s75_I_A1H5L": (31, 69),
    "8s75_L_A1H5L": (31, 69),
    "8s76_C_A1H5M": (36, 72),
    "9c5v_E_A1AUO": (60, 61),
}


def add_hydrogens(mol):
    """Add missing hydrogens, moving only those atoms so deposited coordinates stay exact."""
    n_deposited = mol.GetNumAtoms()
    mol = Chem.AddHs(mol, addCoords=True)
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is not None and mol.GetNumAtoms() > n_deposited:
        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
        for i in range(n_deposited):
            ff.AddFixedPoint(i)
        ff.Initialize()
        ff.Minimize(maxIts=2000)
    return mol


def clash_count(mol):
    """Atom pairs more than three bonds apart that sit well inside their van der Waals contact."""
    pos = mol.GetConformer().GetPositions()
    table = Chem.GetPeriodicTable()
    radii = np.array([table.GetRvdw(a.GetAtomicNum()) for a in mol.GetAtoms()])
    dist = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    close = dist < CLASH_VDW_FRACTION * (radii[:, None] + radii[None])
    return int(np.triu(close & (Chem.GetDistanceMatrix(mol) > 3), 1).sum())


def load_conformers():
    entries = defaultdict(list)
    for path in sorted(SDF_DIR.glob("*.sdf")):
        if path.stem in INCOMPLETE:
            continue
        pdb, _, ligand = path.stem.split("_")
        mol = add_hydrogens(Chem.MolFromMolFile(str(path), removeHs=False))
        entries[pdb].append((clash_count(mol), path.stem, ligand, mol))
    return entries


def wireframe(mol):
    pos = mol.GetConformer().GetPositions()
    pos = (pos - pos.mean(axis=0)) / ANGSTROM_PER_UNIT
    Chem.Kekulize(mol, clearAromaticFlags=True)

    segs = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        pa, pb = pos[a], pos[b]
        segs.append((*pa, *pb))
        if bond.GetBondType() not in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
            continue

        axis = pb - pa
        nbr = next(
            (n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() != b),
            next((n.GetIdx() for n in mol.GetAtomWithIdx(b).GetNeighbors() if n.GetIdx() != a), None),
        )
        ref = pos[nbr] - pa if nbr is not None else np.cross(axis, [0.0, 0.0, 1.0])
        perp = ref - axis * (ref @ axis) / (axis @ axis)
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(axis, [1.0, 0.0, 0.0])
        perp = perp / np.linalg.norm(perp) * DOUBLE_OFFSET
        offsets = [perp] if bond.GetBondType() == Chem.BondType.DOUBLE else [perp, -perp]
        for off in offsets:
            segs.append((*(pa + off + axis * 0.12), *(pb + off - axis * 0.12)))

    elements = [ELEMENT_CLASS.get(at.GetSymbol(), 5) for at in mol.GetAtoms()]
    return np.array(segs), pos, elements


def quantize(arr):
    return np.round(arr * QUANT).astype(int).ravel().tolist()


def main():
    molecules = []
    for pdb, conformers in sorted(load_conformers().items()):
        clashes, source, ligand, mol = min(conformers, key=lambda c: c[:2])
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        segs, atoms, elements = wireframe(mol)
        molecules.append(
            {
                "pdb": pdb,
                "ligand": ligand,
                "source": source,
                "smiles": smiles,
                "s": quantize(segs),
                "a": quantize(atoms),
                "e": elements,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"scale": QUANT, "molecules": molecules}, separators=(",", ":")))
    print(f"wrote {len(molecules)} molecules -> {OUT.relative_to(ROOT)} "
          f"({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
