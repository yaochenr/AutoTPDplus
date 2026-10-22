












import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data" / "release"
OUT = ROOT / "docs" / "assets" / "molecules.json"

SEED = 7
N_PROTAC, N_GLUE = 100, 50
ANGSTROM_PER_UNIT = 1.5
DOUBLE_OFFSET = 0.16
QUANT = 50

ELEMENT_CLASS = {"C": 0, "N": 1, "O": 2, "S": 3, "P": 3, "F": 4, "Cl": 4, "Br": 4, "I": 4}


def load_smiles(csv_name, min_heavy, max_heavy):
    df = pd.read_csv(RELEASE / csv_name, usecols=["SMILES"]).dropna()
    seen, out = set(), []
    for smi in df["SMILES"].unique():
        mol = Chem.MolFromSmiles(smi)
        if mol is None or "." in smi:
            continue
        if not min_heavy <= mol.GetNumHeavyAtoms() <= max_heavy:
            continue
        key = Chem.MolToInchiKey(mol)[:14]
        if key in seen:
            continue
        seen.add(key)
        out.append(smi)
    return out


def specified_centres(mol):
    return {
        i: label
        for i, label in Chem.FindMolChiralCenters(mol, useLegacyImplementation=False)
        if mol.GetAtomWithIdx(i).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    }


def embed(smi, seed):
    ref = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(ref)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
    AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
    mol = Chem.RemoveHs(mol)


    want = specified_centres(ref)
    probe = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(probe)
    got = dict(Chem.FindMolChiralCenters(probe, useLegacyImplementation=False))
    if any(got.get(i) != label for i, label in want.items()):
        return None
    return mol


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
    rng = random.Random(SEED)
    pools = {
        "protac": load_smiles("AutoTPDplus-PROTAC.csv", 35, 90),
        "glue": load_smiles("AutoTPDplus-MG.csv", 15, 60),
    }
    wanted = {"protac": N_PROTAC, "glue": N_GLUE}

    molecules = []
    for kind, pool in pools.items():
        rng.shuffle(pool)
        added = 0
        for smi in pool:
            if added == wanted[kind]:
                break
            mol = embed(smi, SEED)
            if mol is None:
                continue
            segs, atoms, elements = wireframe(mol)
            molecules.append(
                {"kind": kind, "smiles": smi, "s": quantize(segs), "a": quantize(atoms), "e": elements}
            )
            added += 1
        print(f"{kind}: {added} / {wanted[kind]} (pool {len(pool)})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"scale": QUANT, "molecules": molecules}, separators=(",", ":")))
    print(f"wrote {len(molecules)} molecules -> {OUT.relative_to(ROOT)} "
          f"({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
