import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from tpd_curator.preprocessing.smiles_standardization import (
    add_compound_key,
    canonicalize_smiles_column,
)

HUMAN = 9606

UNMAPPED: dict[str, set[str]] = {}
UNPARSED: dict[str, dict[str, str]] = {}

_ENGINEERING_HINT = re.compile(
    r"HiBiT|nLuc|NanoLuc|luciferase|Luc\b|GFP|mCherry|HaloTag|dTAG|FKBP|reporter|"
    r"knock[- ]?(?:in|out|down)|\bKO\b|CRISPR|Cas9|(?-i:sh[A-Z0-9])|(?-i:si[A-Z0-9])|-/-|"
    r"overexpress|express(?:ing|ed)|silenc|transfect|transduc|stabl[ey]|tagged|fusion",
    re.IGNORECASE,
)


def _note_unmapped(kind: str, value) -> None:
    UNMAPPED.setdefault(kind, set()).add(str(value))


def _note_unparsed(field: str, raw, *, recognised: bool = False) -> None:
    UNPARSED.setdefault(field, {}).setdefault(str(raw), "recognised" if recognised else "unparsed")

DEFAULT_INPUT_CSV = "data/protacs/llm/protac-data-llm.csv"
DEFAULT_OUTPUT_CSV = "data/AutoTPD-PROTAC.csv"


def _place_after(df: pd.DataFrame, anchor: str, followers: list[str]) -> pd.DataFrame:
    followers = [c for c in followers if c in df.columns]
    if anchor not in df.columns or not followers:
        return df
    cols = [c for c in df.columns if c not in followers]
    idx = cols.index(anchor)
    new_cols = cols[: idx + 1] + followers + cols[idx + 1:]
    return df[new_cols]


def _normalize_species(label: str | None) -> str | None:
    if not label:
        return None
    return re.sub(r"\s*\([^()]*\)\s*$", "", label).strip() or None


def cellosaurus_search(query: str) -> dict | None:
    url = "https://api.cellosaurus.org/search/cell-line"
    try:
        r = requests.get(url, params={
            "q": f"id:{query}",
            "format": "json",
            "fields": "ac,id,sy,ox",
        }, timeout=15)
        if not r.ok:
            return None
        results = r.json().get("Cellosaurus", {}).get("cell-line-list", [])
        if not results:
            return None
        for entry in results:
            names = entry.get("name-list", [])
            rec_name = None
            for n in names:
                if n.get("type") == "identifier":
                    rec_name = n["value"]
                    break
            if rec_name and rec_name.lower() == query.lower():
                acc = None
                for a in entry.get("accession-list", []):
                    if a.get("type") == "primary":
                        acc = a["value"]
                        break
                species = None
                taxonomy_id = None
                sp_list = entry.get("species-list", [])
                if sp_list:
                    species = _normalize_species(sp_list[0].get("label"))
                    taxonomy_id = int(sp_list[0].get("accession", 0)) or None
                return {"accession": acc, "name": rec_name, "species": species, "taxonomy_id": taxonomy_id}
    except Exception as e:
        print(f"  Cellosaurus error for '{query}': {e}")
    return None


def cellosaurus_fetch(accession: str) -> dict | None:
    url = f"https://api.cellosaurus.org/cell-line/{accession}"
    try:
        r = requests.get(url, params={"format": "json", "fields": "ac,id,sy,ox"}, timeout=15)
        if not r.ok:
            return None
        entry = r.json().get("Cellosaurus", {}).get("cell-line-list", [{}])[0]
        names = entry.get("name-list", [])
        rec_name = next((n["value"] for n in names if n["type"] == "identifier"), None)
        species = None
        taxonomy_id = None
        sp_list = entry.get("species-list", [])
        if sp_list:
            species = _normalize_species(sp_list[0].get("label"))
            taxonomy_id = int(sp_list[0].get("accession", 0)) or None
        return {"accession": accession, "name": rec_name, "species": species, "taxonomy_id": taxonomy_id}
    except Exception as e:
        print(f"  Cellosaurus fetch error for '{accession}': {e}")
    return None

CL_SEARCH_MAP = {
    "22RV1":                "22Rv1",
    "293T":                 "HEK293T",
    "A375":                 "A-375",
    "A375 xenograft":       "A-375",
    "A549":                 "A-549",
    "Ba/F3 Bcr-AblT315I":  "Ba/F3",
    "BMDMs":                None,
    "CAL-12-T":             "CAL-12T",
    "CAOV4":                "Caov-4",
    "Flip-In 293":          "Flp-In-293",
    "Flp-In T-REx 293":    "Flp-In-T-REx-293",
    "Flp293T GFP-GSPT1(389-499)/mCherry": "HEK293T",
    "G2019S LRRK2 MEFs":   None,
    "H1666":                "NCI-H1666",
    "H293T":                "HEK293T",
    "HCC-15":               "HCC15",
    "HCC-364 vr1":          "HCC364",
    "HCT116":               "HCT 116",
    "HCT15":                "HCT 15",
    "HEK-293T (BTK-nLuc)":  "HEK293T",
    "HEK-293T (IKZF1-nLuc)":"HEK293T",
    "HEK-293T (IKZF3-nLuc)":"HEK293T",
    "HEK293 (α1A-AR stably transfected)": "HEK293",
    "HEK293 BRD9-HiBiT/CAS9/FF":  "HEK293",
    "HEK293 BRD9-HiBiT/FF/CAS9":  "HEK293",
    "HEK293-hTau":          "HEK293",
    "HEK293/BRD9-HiBiT":   "HEK293",
    "HepG2":                "Hep-G2",
    "Hs746T":               "Hs 746.T",
    "Insig-silenced HepG2": "Hep-G2",
    "Jurkat E6-1":          "Jurkat E6.1",
    "K562":                 "K-562",
    "K562 HiBiT-tagged":    "K-562",
    "MKN-45":               "MKN45",
    "MM.1S":                "MM1.S",
    "MOLT4":                "MOLT-4",
    "MOLT4 HiBiT-IKZF1":   "MOLT-4",
    "MV-4-11":              "MV4-11",
    "MV4;11":               "MV4-11",
    "NIH-3T3":              "NIH 3T3",
    "Panc Tu-I":            "PancTu-I",
    "R1441C LRRK2 MEFs":   None,
    "Rama 37 (S100A4-transfected)": "Rama 37",
    "SK-MEL-239 C4":        "SK-MEL-239",
    "SK-N-BE(2)-C":         "CVCL_0529",
    "TMD-8":                "TMD8",
    "TMD8 BTK-GFP/mCh":    "TMD8",
    "U87":                  "U-87MG ATCC",
    "WI38 non-senescent (NCs)": "WI-38",
    "WT LRRK2 MEFs":        None,

    "BxPC3":                "BxPC-3",
    "CWR22Rv1 (CRISPR HiBiT knock-in)": "22Rv1",
    "HEK293T–Cas9":         "HEK293T",
    "KBM7":                 "KBM-7",
    "Karpas 422":           "Karpas-422",
    "MOLT4 CK1α-HiBiT":    "MOLT-4",
    "primary human Tregs":  None,

    "BE(2)-C":              "CVCL_0529",
    "Human PBMCs":          None,
    "Human monocytes (n=3)": None,
    "Human whole blood":    None,
    "JAR":                  "JAR",
    "Mouse primary splenocytes": None,
    "OCI-Ly1":              "OCI-Ly1",
    "OCI-Ly10":             "OCI-Ly10",
    "OCI-Ly11":             "OCI-Ly11",
    "OCILY3":               "OCI-Ly3",
    "PA-1":                 "PA-1",
    "SU-DHL-4":             "SU-DHL-4",
    "hMDM":                 None,
    "primary human erythroblasts": None,

    "22Rv1":                "22Rv1",
    "231MFP":               "MDA-MB-231",
    "293FT":                "HEK293-FT",
    "4T1":                  "4T1",
    "A-204":                "A-204",
    "A152T neurons":        None,
    "A2780":                "A2780",
    "A431":                 "A-431",
    "A549/Taxol":           "A-549",
    "AML12":                "AML12",
    "AsPC-1":               "AsPC-1",
    "BT474":                "BT-474",
    "Ba/F3":                "Ba/F3",
    "BaF3 FLT3-ITD":        "Ba/F3",
    "Bel-7402":             "BEL-7402",
    "CA-46":                "CA46",
    "CAL-12T":              "CAL-12T",
    "CAL-148":              "CAL-148",
    "CAL33":                "CAL-33",
    "CCLP1":                "CC-LP-1",
    "CHL-1":                "CHL-1",
    "COLO 205":             "COLO 205",
    "COLO-205":             "COLO 205",
    "CWR22Rv1":             "22Rv1",
    "Caki-1":               "Caki-1",
    "Calu-1":               "Calu-1",
    "Calu-6":               "Calu-6",
    "Capan-1":              "Capan-1",
    "DB":                   "DB",
    "DOHH2":                "DOHH2",
    "Dermal fibroblast":    None,
    "EBC-1":                "EBC-1",
    "EOL-1":                "EoL-1",
    "ER-positive breast cancer cells": None,
    "GBM43":                "GBM43",
    "Germ cells":           None,
    "H1650":                "NCI-H1650",
    "H1650R":               "NCI-H1650",
    "H1975":                "NCI-H1975",
    "H23":                  "NCI-H23",
    "H3122":                "NCI-H3122",
    "H3255":                "NCI-H3255",
    "H358":                 "NCI-H358",
    "H838":                 "NCI-H838",
    "HAP1":                 "HAP1",
    "HBL-1":                "HBL-1 [Human diffuse large B-cell lymphoma]",
    "HCC-827":              "HCC827",
    "HCC1937":              "HCC1937",
    "HCC827":               "HCC827",
    "HCT-116":              "HCT 116",
    "HD-MB03":              "HD-MB03",
    "HEI-OC1":              "HEI-OC1",
    "HEK293":               "HEK293",
    "HEK293/BRD7-HiBiT":    "HEK293",
    "HEK293A":              "HEK293-A",
    "HEK293T":              "HEK293T",
    "HEL":                  "HEL",
    "HGC-27":               "HGC-27",
    "HGC-27/HUVEC":         None,
    "HL-60":                "HL-60",
    "HL60":                 "HL-60",
    "HLE":                  "HLE",
    "HLF":                  "HLF",
    "HT-1080":              "HT-1080",
    "HT-29":                "HT-29",
    "HT1080":               "HT-1080",
    "HT29":                 "HT-29",
    "HUCCT1":               "HuCC-T1",
    "HUH-1":                "HuH-1",
    "HeLa":                 "HeLa",
    "Hep3B2.1-7":           "Hep 3B2.1-7",
    "Hs578t":               "Hs 578T",
    "HuH-7":                "HuH-7",
    "Huh7.5":               "Huh-7.5",
    "IgE MM":               None,
    "JeKo-1":               "JeKo-1",
    "Jeko-1":               "JeKo-1",
    "Jurkat":               "Jurkat",
    "KATO III":             "KATO III",
    "KCL22":                "KCL-22",
    "KM12":                 "KM12",
    "KOPN49":               "KOPN-49",
    "KOPT-K1":              "KOPT-K1",
    "KU182":                None,
    "KU812":                "KU812",
    "KU812 CML":            "KU812",
    "KYSE-270":             "KYSE-270",
    "KYSE520":              "KYSE-520",
    "Karpas 299":           "Karpas-299",
    "Karpas299":            "Karpas-299",
    "Kasumi-1":             "Kasumi-1",
    "Kelly":                "Kelly",
    "L-O2":                 "L-02",
    "LCC2":                 "CVCL_DP51",
    "LNCaP":                "LNCaP",
    "LnCaP95":              "LNCaP-95",
    "MCF-7":                "MCF-7",
    "MCF7":                 "MCF-7",
    "MDA-MB-231":           "MDA-MB-231",
    "MDA-MB-436":           "MDA-MB-436",
    "MDA-MB-453":           "MDA-MB-453",
    "MDA-MB-468":           "MDA-MB-468",
    "MDA-Pca-2b":           "MDA-PCa-2b",
    "MEFs":                 "MEF",
    "MHH-CALL-4":           "MHH-CALL-4",
    "MIA PaCa-2":           "MIA PaCa-2",
    "MOLM-13":              "MOLM-13",
    "MOLM-14":              "MOLM-14",
    "MOLM-16":              "MOLM-16",
    "MOLM13":               "MOLM-13",
    "MOLT-4":               "MOLT-4",
    "MPro-eGFP stable cells": None,
    "MV4-11":               "MV4-11",
    "Mino":                 "Mino",
    "Molm-16":              "MOLM-16",
    "Molt-4":               "MOLT-4",
    "Mouse 4935":           None,
    "NAMALWA":              "Namalwa",
    "NB4":                  "NB4",
    "NCI-H1568":            "NCI-H1568",
    "NCI-H1703":            "NCI-H1703",
    "NCI-H1975":            "NCI-H1975",
    "NCI-H2030":            "NCI-H2030",
    "NCI-H2228":            "NCI-H2228",
    "NCI-H23":              "NCI-H23",
    "NCI-H358":             "NCI-H358",
    "NCI-H661":             "NCI-H661",
    "NCI-H838":             "NCI-H838",
    "NGP":                  "NGP",
    "Namalwa":              "Namalwa",
    "OCI-AML2":             "OCI-AML-2",
    "OCI-AML3":             "OCI-AML-3",
    "OVCAR-5":              "OVCAR-5",
    "OVCAR8":               "OVCAR-8",
    "PA1":                  "PA-1",
    "PANC1":                "PANC-1",
    "PBMC":                 None,
    "PC-3":                 "PC-3",
    "PC3":                  "PC-3",
    "PC3-S1":               "PC-3",
    "PC9":                  "PC-9",
    "PDX SJBALL020589":     None,
    "Pfeiffer":             "Pfeiffer",
    "Primary Cardiomyocytes": None,
    "Primary Sertoli cells": None,
    "RAMOS":                "Ramos",
    "RAW 264.7":            "RAW 264.7",
    "RI-1":                 "RI-1",
    "RKO":                  "RKO",
    "RS4;11":               "RS4;11",
    "Raji":                 "Raji",
    "Ramos":                "Ramos",
    "SH-SY5Y":              "SH-SY5Y",
    "SK-Hep-1":             "SK-HEP-1",
    "SK-LU-1":              "SK-LU-1",
    "SK-MEL-246":           "SK-MEL-246",
    "SK-MEL-28":            "SK-MEL-28",
    "SK-MEL-5":             "SK-MEL-5",
    "SKNO1":                "SKNO-1",
    "SNU-387":              "SNU-387",
    "SNU-398":              "SNU-398",
    "SNU-423":              "SNU-423",
    "SR":                   "SR",
    "SRD15":                "SRD-15",
    "SU-DHL-1":             "SU-DHL-1",
    "SUM149":               "SUM149PT",
    "SUP-M2":               "SUP-M2",
    "SW1573":               "SW1573",
    "SW480":                "SW480",
    "SW620":                "SW620",
    "Sk-Mel-28":            "SK-MEL-28",
    "Snca OE-PFF seeding HEK293T": "HEK293T",
    "T-cell leukemia Jurkat": "Jurkat",
    "T47D":                 "T-47D",
    "THP-1":                "THP-1",
    "THP1":                 "THP-1",
    "TM3":                  "CVCL_4326",
    "TMD8":                 "TMD8",
    "U251":                 "U-251MG",
    "U937":                 "U-937",
    "VCaP":                 "VCaP",
    "VCaP AR+":             "VCaP",
    "WI38":                 "WI-38",
    "XLA":                  None,
    "Z138":                 "Z-138",
    "hPBMC":                None,
    "human dermal papilla": None,
    "platelets":            None,

    "4xSNCA iPSC-derived dopaminergic neurons (DANs)": None,
    "EGFR-AXL chimeric cell line":   None,
    "EGFR-MERTK chimeric cell line": None,
    "EGFR-TYRO3 chimeric cell line": None,
    "FTD19-L5-RC6 (tau-A152T)":      None,
    "MGH2046-RC1 (tau-P301L)":       None,
    "PBMCs":                 None,
    "Q175 mouse cortical neurons":   None,
    "human PBMCs":           None,
    "mouse BMDMs (naïve)":   None,
    "mouse primary Sertoli cells (C57BL/6N, 6 dpp)": None,
    "mouse primary germ cells (C57BL/6N, 6 dpp)":    None,
    "murine peritoneal macrophages":  None,
    "primary CLL cells":     None,
    "primary rat neonatal cardiomyocytes": None,
    "tau-P301L iPSC-derived neurons": None,

    "HeLa cells":                            "HeLa",
    "Jurkat E6.1":                           "Jurkat E6.1",
    "Jurkat +elacridar":                     "Jurkat",
    "Jurkat -elacridar":                     "Jurkat",
    "Neuro-2a +elacridar":                   "Neuro-2a",
    "Neuro-2a -elacridar":                   "Neuro-2a",
    "SH-SY5Y neuroblastoma +elacridar":      "SH-SY5Y",
    "SH-SY5Y neuroblastoma -elacridar":      "SH-SY5Y",
    "TMD8 BTK-GFP/mCherry":                  "TMD8",
    "Q175 mouse cortical neuron":            None,
    "Q175 mouse cortical neuron -elacridar": None,
    "primary cells from chronic lymphocytic leukemia (CLL) patients": None,
    "293T GSPT1-HiBiT":                    "HEK293T",
    "293T HBS1L-HiBiT":                    "HEK293T",
    "A549 (HiBiT-CCNK KI)":                "A-549",
    "Activated primary human CD8+ T cells": None,
    "B16F10":                               "B16-F10",
    "CK1α-HiBiT MOLT4":                    "MOLT-4",
    "Colo205":                              "COLO 205",
    "CT26":                                 "CT26",
    "CWR22Rv1 (AR-HiBiT knock-in)":         "22Rv1",
    "dCK-IKZF1-HEK293FT":                   "HEK293-FT",
    "DLD-1 CRBN KO cells stably expressing FLAG-CRBN WT": "DLD-1",
    "DLD-1 WT":                             "DLD-1",
    "DLD1":                                 "DLD-1",
    "Fadu":                                 "FaDu",
    "HCT116 (HiBiT-c-Myc)":                 "HCT 116",
    "HEK293 (CK1α-HiBiT)":                 "HEK293",
    "HEK293 (LgBiT) HBS1L-HiBiT":           "HEK293",
    "HEK293 BRD9-HiBiT KI":                 "HEK293",
    "HEK293 HiBiT knock-in":                "HEK293",
    "HEK293T (293T)":                       "HEK293T",
    "HEK293T Keap1WT":                      "HEK293T",
    "HEK293T SMARCA4–HiBiT":                "HEK293T",
    "HEK293T VHLWT":                        "HEK293T",
    "Hela":                                 "HeLa",
    "Hep3B":                                "Hep 3B2.1-7",
    "HiBiT-BRD4 HEK293 cells":              "HEK293",
    "Jurkat (IKZF1-HiBiT)":                 "Jurkat",
    "Jurkat VAV1-HiBiT knock-in":           "Jurkat",
    "KBM7 (BRD4Tandem reporter)":           "KBM-7",
    "KOPT-K1_LCK_HiBiT":                    "KOPT-K1",
    "KYSE30":                               "KYSE-30",
    "KYSE450":                              "KYSE-450",
    "MDA-MB-175VII":                        "MDA-MB-175-VII",
    "MIA PaCa-2 ZBTB11-HiBiT knock-in":     "MIA PaCa-2",
    "NCI-H929":                             "NCI-H929",
    "NIH 3T3":                              "NIH 3T3",
    "SK-OV-3":                              "SK-OV-3",
    "U2OS NSD2 HiBiT":                      "U2OS",
    "U87MG":                                "U-87MG ATCC",
    "UMRC2":                                "UM-RC-2",
    "UMRC2 ALG12 KO":                       "UM-RC-2",
    "UMRC2 GNPTAB KO":                      "UM-RC-2",
    "UMRC2 VPS26A KO":                      "UM-RC-2",
    "UMRC2 WT":                             "UM-RC-2",
    "unspecified (DEFUSE system)":           None,
    "various AML cell lines (HiBiT–ENL reporter)": None,
    "2208L": None,
    "22Rv1 cells": "22Rv1",
    "231MFP human breast cancer": "MDA-MB-231",
    "293 T": "HEK293T",
    "293T GFP-KRAS": "HEK293T",
    "293T cells": "HEK293T",
    "293T stably transfected with Mpro-GFP": "HEK293T",
    "4935": None,
    "4935 murine cells": None,
    "4T1cl5": "4T1",
    "5 KRAS G12C-mutant cell lines": None,
    "528 GSCs": None,
    "6 T-CEM": "6T-CEM",
    "786–0": "786-O",
    "A2780, CT26": None,
    "A2780, HUVEC": None,
    "A427": "A-427",
    "A498": "A-498",
    "A549 (ED-tagged KRAS(G12C))": "A-549",
    "A549 clone 25": "A-549",
    "A549 clone 26": "A-549",
    "A549 clone 27": "A-549",
    "A549, HEK293": None,
    "A704": "A-704",
    "AML cells": None,
    "AML-12": "AML12",
    "AML12 cells": "AML12",
    "AR+ PCa cell lines": None,
    "AR+ prostate cancer cell lines": None,
    "AR+ prostate cancer cells": None,
    "AR-HiBiT LNCaP cells": None,
    "AR-positive prostate cancer cell lines": None,
    "B-cell lymphoma cells": None,
    "B16.F10": "B16-F10",
    "B16–F10": "B16-F10",
    "B6J-S1 mouse embryonic stem cells": None,
    "BEAS-2B cells": "BEAS-2B",
    "BMCs": None,
    "BMDC": None,
    "BRD4HiBiT (HEK293)": "HEK293",
    "BT549": "BT-549",
    "BTK ELISA Kit": None,
    "BTK-HiBiT Ramos": "Ramos",
    "BTK-nLuc Ramos": "Ramos",
    "BV173": "BV-173",
    "BV2 + Halotag": "BV-2",
    "Ba/F3 (EGFRL858R)": "Ba/F3",
    "Ba/F3 CD74-ROS1": "Ba/F3",
    "Ba/F3T315I": "Ba/F3",
    "BaF3 (FLT3-ITD)": "Ba/F3",
    "BaF3-EML4-ALK": "Ba/F3",
    "BromoTag-Brd2 HEK293": "HEK293",
    "Burkitt's lymphoma cell lines": None,
    "Burkitt's lymphoma cells": None,
    "C33A": "C-33 A",
    "C33A, HeLa, CD3  T cells": None,
    "C4-2": "LNCaP C4-2",
    "CAL51": "CAL-51",
    "CCRF": "CCRF-CEM",
    "CT26, B16-F10, A549": None,
    "CWR22rv1": "22Rv1",
    "CellTiter-Glo Kit": None,
    "DHL6": "SU-DHL-6",
    "DLBCL cell lines": None,
    "EL4 (mouse T cell lymphoma)": "EL4",
    "EO771 (mouse mammary cancer)": "EO771",
    "EO771, MCA-205, MC38, HT29, LIM1215": None,
    "EOL-1 AML": "EoL-1",
    "ER+ breast cancer cell lines": None,
    "ER+ breast cancer cell lines (unspecified)": None,
    "EWS/FLI cells": None,
    "Emu-myc": None,
    "Expi293": "Expi293F",
    "FLT3-ITD mutant AML cells": None,
    "FTD19-L5-RC6": None,
    "Flp-In 293 T-REx cells": "Flp-In-T-REx-293",
    "Flp-In T-REx 293 (stable NS3-eGFP-2A-mCherry)": "Flp-In-T-REx-293",
    "Flp293T": "HEK293T",
    "GBM34": None,
    "GSC 528": None,
    "GSC X01": None,
    "GSK3β-HiBiT KI HEK293 cells": "HEK293",
    "H1299": "NCI-H1299",
    "H1299-p53-R175H": "NCI-H1299",
    "H146": "NCI-H146",
    "H1568": "NCI-H1568",
    "H1792": "NCI-H1792",
    "H1975 (EGFRL858R/T790M)": "NCI-H1975",
    "H1975-TM": "NCI-H1975",
    "H2023": "NCI-H2023",
    "H2030": "NCI-H2030",
    "H211": "NCI-H211",
    "H2126": "NCI-H2126",
    "H2228": "NCI-H2228",
    "H322": "NCI-H322",
    "H441": "NCI-H441",
    "H508": "NCI-H508",
    "HAP1 HiBiT": "HAP1",
    "HAP1-TNKS1-NanoLuc": "HAP1",
    "HBL1": "HBL-1 [Human diffuse large B-cell lymphoma]",
    "HCC 15": "HCC15",
    "HCC 1954": "HCC1954",
    "HCC cell lines (panel of 11)": None,
    "HCC1954 (ER–, KAT6A amp)": "HCC1954",
    "HCC827 (EGFRdel19)": "HCC827",
    "HCC827 (exon 19 del)": "HCC827",
    "HCC‐827": "HCC827",
    "HCT116 (HiBiT-FKBP12F36V-KRASG12D), stably expressing FLAG-MTH1-RNF152": None,
    "HCT116 cells": "HCT 116",
    "HCT116-HiBiT-BRD4": "HCT 116",
    "HDPC": None,
    "HDPCs": None,
    "HEK 293T": "HEK293T",
    "HEK-293": "HEK293",
    "HEK-293 (ALK2-HiBiT overexpression)": "HEK293",
    "HEK-293T": "HEK293T",
    "HEK-293T17 (BTK-nLuc)": "HEK293T/17",
    "HEK-293T17 (IKZF1-nLuc)": "HEK293T/17",
    "HEK-293T17 (IKZF3-nLuc)": "HEK293T/17",
    "HEK293 (BromoTag-Brd2 knock-in)": "HEK293",
    "HEK293 (Flp-In 293)": "HEK293",
    "HEK293 (HiBiT-tagged)": "HEK293",
    "HEK293 (HiBiT–BromoCatch–Brd4 knock-in)": "HEK293",
    "HEK293 (HiBit-BRD4)": "HEK293",
    "HEK293 (METex14Δ-GFP)": "HEK293",
    "HEK293 (endogenous HiBiT-BRD4 knock-in)": "HEK293",
    "HEK293 (endogenous HiBiT–Brd2)": "HEK293",
    "HEK293 (endogenous HiBiT–Brd3)": "HEK293",
    "HEK293 (endogenous HiBiT–Brd4)": "HEK293",
    "HEK293 (human embryonic kidney)": "HEK293",
    "HEK293 (stable 3CLpro expression)": "HEK293",
    "HEK293 (stably transfected with α1A-AR)": "HEK293",
    "HEK293 CRISPR HiBiT BRD4 cell line": "HEK293",
    "HEK293 Flp-In T-REx": "Flp-In-T-REx-293",
    "HEK293 GSK-3β-HiBiT knock-in": "HEK293",
    "HEK293 Halo-VPS34 knock-in (Clone 56)": "HEK293",
    "HEK293 HiBiT-BRD4 Cln3": "HEK293",
    "HEK293 LgBiT stable cells": "HEK293",
    "HEK293 OE Mpro": "HEK293",
    "HEK293 SGK3-Halo knock-in (Clone 2)": "HEK293",
    "HEK293 SNAP-EGFP": "HEK293",
    "HEK293 cells": "HEK293",
    "HEK293 cells stably transfected with Mpro": "HEK293",
    "HEK293 expressing BTK": "HEK293",
    "HEK293, Jurkat T, KB, HeLa": None,
    "HEK293, Jurkat, MEF, U2OS, MC38, MiaPaca2, H116, H358, HepG2, B16F10": None,
    "HEK293-Halo-VPS34": "HEK293",
    "HEK293-HiBiT": "HEK293",
    "HEK293-LgBiT": "HEK293",
    "HEK293-SGK-Halo": "HEK293",
    "HEK293-hTau and SH-SY5Y": "SH-SY5Y",
    "HEK293-hTau overexpressed cell line": "HEK293",
    "HEK293-hTau, SH-SY5Y (p-tau induced using okadaic acid)": None,
    "HEK293-htau": "HEK293",
    "HEK293T (AR-GFP transfected)": "HEK293T",
    "HEK293T (GFP-HaloTag-expressing)": "HEK293T",
    "HEK293T (GFP-NCL-expressing)": "HEK293T",
    "HEK293T (USP39-OFP + VHL_WT-EGFP)": "HEK293T",
    "HEK293T (stably expressing 3CPro-Flag)": "HEK293T",
    "HEK293T BRD4-HiBiT knock-in cells": "HEK293T",
    "HEK293T HiBiT-TEAD1": "HEK293T",
    "HEK293T HiBiT-TEAD2": "HEK293T",
    "HEK293T HiBiT-TEAD3": "HEK293T",
    "HEK293T HiBiT-TEAD4": "HEK293T",
    "HEK293T cells": "HEK293T",
    "HEK293T-NSD2-HiBiT": "HEK293T",
    "HEK293T/17 (SHP2-sfGFPxK stable line)": "HEK293T/17",
    "HEK293TBRD4‑HiBiT": "HEK293T",
    "HEK293_HFNES": "HEK293",
    "HEP3B": "Hep 3B2.1-7",
    "HEPG2": "Hep-G2",
    "HK-2": "HK-2 [Human kidney]",
    "HMEC": None,
    "HN30": "WSU-HN30",
    "HS-SY-II": "HS-SY-2",
    "HSC-T6 cells": "HSC-T6",
    "HT-1080, U937, HT-29, MDA-MB-231, MDA-MB-468, MDA-MB-361, HaCaT, A431, Kym-1, L929, E0771, MC38, MCA-205, EMT6, BMDMs, LFs, MDFs": None,
    "HT29 cells": "HT29",
    "HUVEC": None,
    "HUVECs": None,
    "HUVECs (Ang-II-stimulated)": None,
    "HeLa (HiBiT-KAT6A)": "HeLa",
    "HeLa (HiBiT-SMARCA2)": "HeLa",
    "HeLa (HiBiT-SMARCA4)": "HeLa",
    "HeLa (HttQ97-NES-eGFP)": "HeLa",
    "HeLa (HttQ97-NLS-eGFP)": "HeLa",
    "HeLa (overexpressed mutant)": "HeLa",
    "HeLa (recombinant BTKC481S)": "HeLa",
    "HeLa, stable HA-tagged Mpro": None,
    "HeLa-USP39_OFP": "HeLa",
    "HeLasEH-HiBiT": "HeLa",
    "Hela cells": "HeLa",
    "Hela, MOLM-16, SU-DHL-1, SUP-M2, Pfeiffer, PBMCs": None,
    "HepG2 cells": "Hep-G2",
    "HepG2.2.15; HepAD38": None,
    "HiBiT-BRD4 HEK293 CRISPR cells": "HEK293",
    "HiBiT-BromoTag-BRD4 HEK293": "HEK293",
    "HiBiT-KRASG12D GP5d": "GP5d",
    "HiBiT-KRASG12R Cal-62": "CAL-62",
    "HuCCT1": "HuCC-T1",
    "HuH7": "Huh-7",
    "Huh7": "Huh-7",
    "Huh7-HBP": "Huh-7",
    "Human PBMCs (lymphocytes and monocytes)": None,
    "Human embryonic kidney 293T cells": "HEK293T",
    "Human neuronal cell models derived from a PSP patient carrier of Tau-A152T risk variant and one derived from a FTD patient carrier of Tau P301L": None,
    "Hut78": "HuT 78",
    "IMR32 cells": "IMR-32",
    "Insig-silenced HepG2 cells": "Hep-G2",
    "JVM3": "JVM-3",
    "Jurkat (ITKHiBiT)": "Jurkat",
    "Jurkat (human T cell leukemia)": "Jurkat",
    "Jurkat BRD4-HiBiT cells": "Jurkat",
    "Jurkat E6-1 (+elacridar, 10 μM, 1 h pretreatment)": "Jurkat E6.1",
    "Jurkat E6-1 (−elacridar)": "Jurkat E6.1",
    "Jurkat T, CT-26": None,
    "Jurkat T, HeLa, CT26, PBMC, CAR-T": None,
    "Jurkat T, Human primary T cells": None,
    "Jurkat T, MC38": None,
    "Jurkat cells": "Jurkat",
    "Jurkat nLuc-RIPK1": "Jurkat",
    "Jurkat, MOLT4, HL60, SW620, HT29, HCT116, MCF-7, HEK293T": None,
    "K562 HiBiT": "K-562",
    "K562-Cas9": "K-562",
    "KASUMI1": "Kasumi-1",
    "KBM7 iCas9": "KBM-7",
    "KG-1 (AML)": "KG-1",
    "KG1a": "KG-1a",
    "KMS27": "KMS-27",
    "KPC cells": None,
    "KRAS G12C-expressing cell lines": None,
    "KRAS G12D-HiBiT KI HEK293": "HEK293",
    "KRAS G12V-HiBiT KI SW620": "SW620",
    "KYSE180": "KYSE-180",
    "KYSE520 cells": "KYSE-520",
    "LRRK2G2019S MEF": None,
    "M1-RAW264.7": "RAW 264.7",
    "MC38": "MC-38",
    "MCF-7 WT": "MCF-7",
    "MCF-7 Y537S": "MCF-7",
    "MCF-7 cells": "MCF-7",
    "MCF7 (ESR1-NanoLuc reporter)": "MCF-7",
    "MCF7 CDX": "MCF-7",
    "MCF7 Parental": "MCF-7",
    "MCF7-Neo/HER2": "MCF-7",
    "MC7F": "MCF-7",
    "MDA PCa 2B": "MDA-PCa-2b",
    "MDA-MB-241": "MDA-MB-231",
    "MDA-MB-231 cells": "MDA-MB-231",
    "MDA-MB-231 stem-like cells": "MDA-MB-231",
    "MDA-MB-231 xenograft (BALB/c nude mouse)": "MDA-MB-231",
    "MDA-MB-231, MCF7, T47D, BT-549, MDA-MB-468, HCC1806, MM.1 S, A549, H1299": None,
    "MDA-MB-467": "MDA-MB-468",
    "MDA-MB-468 TNBC cells": "MDA-MB-468",
    "MDA-MB-468 cells": "MDA-MB-468",
    "MDA-MB-468; T47D": None,
    "MDAMB-231": "MDA-MB-231",
    "MGC803": "MGC-803",
    "MGH2046-RC1": None,
    "MM.1S cells": "MM1.S",
    "MM1S": "MM1.S",
    "MM1S cells": "MM1.S",
    "MM1S human cells": "MM1.S",
    "MNNG/HOS": "MNNG/HOS Cl #5",
    "MOLM-3": "MOLM-13",
    "MOLM-16 cells": "MOLM-16",
    "MOLM14": "MOLM-14",
    "MOLT 4": "MOLT-4",
    "MOLT-4 T-ALL cells": "MOLT-4",
    "MOLT-4 cells": "MOLT-4",
    "MOLT4 cells": "MOLT-4",
    "MOLT4; 293T; K562": None,
    "MV-4–11": "MV4-11",
    "MV4-11 (HiBiT-USP7)": "MV4-11",
    "MV4-11 (HiBiT-WDR5)": "MV4-11",
    "MV4-11AURKA-HiBiT": "MV4-11",
    "MV4-11AURKB-HiBiT": "MV4-11",
    "MV4-11AURORA‑A‑HiBiT": "MV4-11",
    "MV4-11AURORA‑A‑Nluc(Kless)": "MV4-11",
    "MV4-11HiBiT-TTK": "MV4-11",
    "MV4 leukemia cell line": "MV4-11",
    "MV4.11": "MV4-11",
    "MV411": "MV4-11",
    "MV4:11": "MV4-11",
    "MV4;11 cells": "MV4-11",
    "MV4–11 AURKAHiBiT": "MV4-11",
    "MYCN-amplified neuroblastoma cells": None,
    "MiaPaCa-2": "MIA PaCa-2",
    "Molt4": "MOLT-4",
    "Msm 607": None,
    "N2a cells": "Neuro-2a",
    "NCI-H2052 (HiBiT-TEAD1)": "NCI-H2052",
    "NCI-H2052 (WT; KO dependency experiment)": "NCI-H2052",
    "NCI-H2052 XIAP KO": "NCI-H2052",
    "NCI-H2052 cIAP1 KO": "NCI-H2052",
    "NCI-H2052 cIAP1/XIAP DKO": "NCI-H2052",
    "NCI-H226 (HiBiT-TEAD1)": "NCI-H226",
    "NCI-H226 (HiBiT-TEAD2)": "NCI-H226",
    "NCI-H226 (HiBiT-TEAD3)": "NCI-H226",
    "NCI-H226 (HiBiT-TEAD4)": "NCI-H226",
    "NCIH1299": "NCI-H1299",
    "NCIH23": "NCI-H23",
    "NCIH460": "NCI-H460",
    "NCIH661": "NCI-H661",
    "NCIH838": "NCI-H838",
    "NCI–H1299": "NCI-H1299",
    "NCI–H2228": "NCI-H2228",
    "NHBE cells": None,
    "NHC-H2030": "NCI-H2030",
    "NHLF": None,
    "NIH/3T3 (mouse embryonic fibroblast)": "NIH 3T3",
    "NT2/D1": "NT2-D1",
    "Neuro-2a (+elacridar, 10 μM, 1 h pretreatment)": "Neuro-2a",
    "Neuro-2a (−elacridar)": "Neuro-2a",
    "Neuro-2a cells": "Neuro-2a",
    "Non-senescent": None,
    "OCI/AML-2": "OCI-AML-2",
    "OVCAR3": "OVCAR-3",
    "PBMC (human)": None,
    "PC-12 cells": "PC-12",
    "PC-3 (FKBP12F36V-EGFP reporter)": "PC-3",
    "PC-3-S1": "PC-3",
    "PC12 cells and PC12 transfected with GFT-tau plasmid": "PC-12",
    "PC‐9": "PC-9",
    "PDX (ESR1 Y537S hormone-independent)": None,
    "PDX cells SJBALL020589": None,
    "PK-59 xenograft": "PK-59",
    "Pan02.13": "Panc02",
    "Panc-1; AsPc-1; MIA-PaCa-2": None,
    "Panc02-OVA": "Panc02",
    "Panc89": "T3M-4",
    "Primary cortical neurons (mouse)": None,
    "Primary rat neonatal cardiomyocytes": None,
    "Prostate cancer cells": None,
    "PyMT-M": None,
    "Q175 knock-in mouse cortical neurons": None,
    "RAW 264.7 macrophage cell line": "RAW 264.7",
    "RAW 264.7 macrophages": "RAW 264.7",
    "RAW264.7": "RAW 264.7",
    "RAW264.7 macrophages": "RAW 264.7",
    "RIPK2 HiBiT HEK293": "HEK293",
    "RPMI 7951": "RPMI-7951",
    "RPMI 8226": "RPMI-8226",
    "RPMI 8226 WT": "RPMI-8226",
    "RS4-11": "RS4;11",
    "RS4-11 leukemia cells": "RS4;11",
    "RS4;11 cells": "RS4;11",
    "RT4": "RT-4",
    "SAEC (healthy)": None,
    "SH-SY5Y (+elacridar)": "SH-SY5Y",
    "SH-SY5Y (−elacridar)": "SH-SY5Y",
    "SH-SY5Y cells": "SH-SY5Y",
    "SH-SY5Y neuronal cells": "SH-SY5Y",
    "SH-SY5Y α-syn A53T": "SH-SY5Y",
    "SH-SY5Y-tauP301L": "SH-SY5Y",
    "SHEP MYCN-ER": "SH-EP",
    "SIBALL020589 PDX": None,
    "SJSA1": "SJSA-1",
    "SK MEL 28": "SK-MEL-28",
    "SK-MEL-28 cells": "SK-MEL-28",
    "SK-N-BE(2)-C neuroblastoma cells": "CVCL_0529",
    "SKBR3": "SK-BR-3",
    "STAT3-HiBiT HeLa": "HeLa",
    "SU-DHL-1 cells": "SU-DHL-1",
    "SUDHL1": "SU-DHL-1",
    "SUM149 cells": "SUM149PT",
    "SUM159": "SUM159PT",
    "Splenocytes": None,
    "T-cell lymphoma cell line": None,
    "T470": "T-47D",
    "T47D cells": "T-47D",
    "T47D, MKN45, RM1, MDA-MB-468, HCC827, MC38, NUGC4, SW579, WM115": None,
    "TCMK1": "TCMK-1",
    "TK1": "CVCL_3217",
    "THP-1 (PMA-differentiated macrophages)": "THP-1",
    "THP-1 cells": "THP-1",
    "TMD-231": "MDA-MB-231",
    "TNBC": None,
    "TNBC MDA-MB-231": "MDA-MB-231",
    "TOV21G": "TOV-21G",
    "Tau overexpressed cell lines (SH-SY5Y, N2a and PC-12)": None,
    "Tau preformed fibril-seeding HEK293T": "HEK293T",
    "Tau-EGFP-overexpressed N2a cells, tau-overexpressed SH-SY5Y cells, 3×Tg-AD mouse primary neuron cells": None,
    "U-251, MM1.S, AU565, A549": None,
    "U-87 MG": "U-87MG ATCC",
    "U251 (overexpressing α-synA53T)": "U-251MG",
    "U2OS KLHDC2 KO/rescue": "U2OS",
    "U87, GBM43, GBM6, CD18, PANC-1, OVCAR5, SKOV3, PC3, MDA-MB-231, GL261-IDO1-O/E, PBMCs": None,
    "U87-MG": "U-87MG ATCC",
    "UM UC-3": "UM-UC-3",
    "WERI-Rb1": "WERI-Rb-1",
    "WI38 (non-senescent)": "WI-38",
    "WI38 non-senescent cells": "WI-38",
    "X01 GSCs": None,
    "XLA (BTK-null)": None,
    "Y79": "Y-79",
    "ZR 75-1": "ZR-75-1",
    "articular chondrocytes": None,
    "canine PBMCs": None,
    "cell-free": None,
    "cynomolgus PBMCs": None,
    "dermal fibroblasts": None,
    "epididymis (mouse tissue)": None,
    "hBMSCs": None,
    "human AML cell line": None,
    "human MM1S cells": "MM1.S",
    "human PBMC": None,
    "human PRP": None,
    "human T cells": None,
    "human monocytes": None,
    "human platelets": None,
    "human primary FLSs": None,
    "human skin fibroblasts": None,
    "mTECs": None,
    "mantle cell lymphoma and DLBCL cells": None,
    "mnng/hos": "MNNG/HOS Cl #5",
    "mouse bone marrow-derived cells (BMDCs, CDK2dTAG)": None,
    "mouse bone marrow-derived cells (BMDCs, CDK5dTAG)": None,
    "mouse duodenum organoids (CDK2dTAG)": None,
    "mouse duodenum organoids (CDK5dTAG)": None,
    "mouse embryonic fibroblast (MEF) cells": None,
    "mouse embryonic stem cells (Crbn+/+)": None,
    "mouse embryonic stem cells (CrbnI391V/I391V)": None,
    "mouse peritoneal macrophages": None,
    "mouse primary cortical neurons": None,
    "mouse primary splenocytes": None,
    "multiple KRASG12C mutant cell lines": None,
    "multiple cell types": None,
    "naive BMDMs (mouse)": None,
    "naive human T cells": None,
    "not specified": None,
    "p53-R175H-expressing cancer cells": None,
    "peripheral blood mononuclear cells": None,
    "preputial gland (mouse tissue)": None,
    "primary Sertoli cells (mouse)": None,
    "primary germ cells (mouse)": None,
    "primary human B cells": None,
    "primary human FAP CAR-eDHFR T cells": None,
    "prostate cancer cell lines": None,
    "prostate cancer cells": None,
    "rat primary splenocytes": None,
    "seminal vesicle (mouse tissue)": None,
    "skin fibroblasts": None,
    "tau-A152T iPSC-derived neurons": None,
    "testis (mouse tissue)": None,
    "unspecified": None,
    "various cancer cell lines": None,
    "various cell lines": None,
    "various human and mouse cell lines": None,
    "washed human platelets": None,
    "ΔG3BP1 + HaloG3BP1": None,
    "ΔG3BP1 + Halotag": None,
}

CL_STD_OVERRIDE: dict[str, str] = {
    "FTD19-L5-RC6 (tau-A152T)":              "FTD19-L5-RC6",
    "G2019S LRRK2 MEFs":                     "MEFs",
    "R1441C LRRK2 MEFs":                     "MEFs",
    "WT LRRK2 MEFs":                         "MEFs",
    "MGH2046-RC1 (tau-P301L)":               "MGH2046-RC1",
    "TM3":                                   "TM3",
}

CL_MODIFICATION_MAP = {

    "A549/Taxol":                              "Taxol-resistant derivative",
    "HCC-364 vr1":                             "vr1",

    "CWR22Rv1 (CRISPR HiBiT knock-in)":   "CRISPR HiBiT knock-in",
    "HEK-293T (BTK-nLuc)":                "BTK-nLuc",
    "HEK-293T (IKZF1-nLuc)":              "IKZF1-nLuc",
    "HEK-293T (IKZF3-nLuc)":              "IKZF3-nLuc",
    "HEK293 BRD9-HiBiT/CAS9/FF":          "BRD9-HiBiT/Cas9/firefly luciferase",
    "HEK293 BRD9-HiBiT/FF/CAS9":          "BRD9-HiBiT/Cas9/firefly luciferase",
    "HEK293/BRD7-HiBiT":                  "BRD7-HiBiT",
    "HEK293/BRD9-HiBiT":                  "BRD9-HiBiT",
    "K562 HiBiT-tagged":                  "HiBiT-tagged",
    "MOLT4 CK1α-HiBiT":                   "CK1α-HiBiT",
    "MOLT4 HiBiT-IKZF1":                  "HiBiT-IKZF1",
    "TMD8 BTK-GFP/mCh":                   "BTK-GFP/mCherry",
    "TMD8 BTK-GFP/mCherry":               "BTK-GFP/mCherry",
    "Flp293T GFP-GSPT1(389-499)/mCherry": "GFP-GSPT1(389-499)/mCherry",
    "WT LRRK2 MEFs":                        "WT LRRK2",

    "FTD19-L5-RC6 (tau-A152T)":              "tau-A152T",
    "MGH2046-RC1 (tau-P301L)":               "tau-P301L",

    "Ba/F3 Bcr-AblT315I":                 "BCR-ABL1 T315I",
    "BaF3 FLT3-ITD":                      "FLT3-ITD",
    "HEK293 (α1A-AR stably transfected)": "α1A-AR stably transfected",
    "HEK293-hTau":                        "human tau (hTau)",
    "Rama 37 (S100A4-transfected)":       "S100A4-transfected",
    "Snca OE-PFF seeding HEK293T":        "SNCA overexpression",
    "SK-MEL-239 C4":                       "C4 clone",

    "G2019S LRRK2 MEFs":                  "LRRK2 G2019S",
    "R1441C LRRK2 MEFs":                  "LRRK2 R1441C",
    "Insig-silenced HepG2":               "INSIG-silenced",
    "293T GSPT1-HiBiT":                    "GSPT1-HiBiT",
    "293T HBS1L-HiBiT":                    "HBS1L-HiBiT",
    "A549 (HiBiT-CCNK KI)":                "HiBiT-CCNK knock-in",
    "CK1α-HiBiT MOLT4":                    "CK1α-HiBiT",
    "CWR22Rv1 (AR-HiBiT knock-in)":         "AR-HiBiT knock-in",
    "dCK-IKZF1-HEK293FT":                   "dCK-IKZF1",
    "DLD-1 CRBN KO cells stably expressing FLAG-CRBN WT": "CRBN knockout; stably expressing FLAG-CRBN WT",
    "DLD-1 WT":                             "WT",
    "HCT116 (HiBiT-c-Myc)":                 "HiBiT-c-Myc",
    "HEK293 (CK1α-HiBiT)":                 "CK1α-HiBiT",
    "HEK293 (LgBiT) HBS1L-HiBiT":           "LgBiT; HBS1L-HiBiT",
    "HEK293 BRD9-HiBiT KI":                 "BRD9-HiBiT knock-in",
    "HEK293 HiBiT knock-in":                "HiBiT knock-in",
    "HEK293T Keap1WT":                      "KEAP1 WT",
    "HEK293T SMARCA4–HiBiT":                "SMARCA4-HiBiT",
    "HEK293T VHLWT":                        "VHL WT",
    "HiBiT-BRD4 HEK293 cells":              "BRD4-HiBiT",
    "Jurkat (IKZF1-HiBiT)":                 "IKZF1-HiBiT",
    "Jurkat VAV1-HiBiT knock-in":           "VAV1-HiBiT knock-in",
    "KBM7 (BRD4Tandem reporter)":           "BRD4 tandem reporter",
    "KOPT-K1_LCK_HiBiT":                    "LCK-HiBiT",
    "MIA PaCa-2 ZBTB11-HiBiT knock-in":     "ZBTB11-HiBiT knock-in",
    "U2OS NSD2 HiBiT":                      "NSD2-HiBiT",
    "UMRC2 ALG12 KO":                       "ALG12 knockout",
    "UMRC2 GNPTAB KO":                      "GNPTAB knockout",
    "UMRC2 VPS26A KO":                      "VPS26A knockout",
    "UMRC2 WT":                             "WT",
    "various AML cell lines (HiBiT–ENL reporter)": "HiBiT-ENL reporter",
    "293T GFP-KRAS": "GFP-KRAS",
    "293T stably transfected with Mpro-GFP": "stably transfected with Mpro-GFP",
    "A549 (ED-tagged KRAS(G12C))": "ED-tagged KRAS(G12C)",
    "AR-HiBiT LNCaP cells": "AR-HiBiT",
    "BRD4HiBiT (HEK293)": "BRD4HiBiT",
    "BTK-HiBiT Ramos": "BTK-HiBiT",
    "BTK-nLuc Ramos": "BTK-nLuc",
    "BV2 + Halotag": "Halotag",
    "Flp-In T-REx 293 (stable NS3-eGFP-2A-mCherry)": "stable NS3-eGFP-2A-mCherry",
    "GSK3β-HiBiT KI HEK293 cells": "GSK3β-HiBiT KI",
    "HAP1 HiBiT": "HiBiT",
    "HAP1-TNKS1-NanoLuc": "TNKS1-NanoLuc",
    "HCT116 (HiBiT-FKBP12F36V-KRASG12D), stably expressing FLAG-MTH1-RNF152": "HiBiT-FKBP12F36V-KRASG12D",
    "HCT116-HiBiT-BRD4": "HiBiT-BRD4",
    "HEK-293 (ALK2-HiBiT overexpression)": "ALK2-HiBiT overexpression",
    "HEK-293T17 (BTK-nLuc)": "BTK-nLuc",
    "HEK-293T17 (IKZF1-nLuc)": "IKZF1-nLuc",
    "HEK-293T17 (IKZF3-nLuc)": "IKZF3-nLuc",
    "HEK293 (BromoTag-Brd2 knock-in)": "BromoTag-Brd2 knock-in",
    "HEK293 (HiBiT-tagged)": "HiBiT-tagged",
    "HEK293 (HiBiT–BromoCatch–Brd4 knock-in)": "HiBiT–BromoCatch–Brd4 knock-in",
    "HEK293 (HiBit-BRD4)": "HiBit-BRD4",
    "HEK293 (METex14Δ-GFP)": "METex14Δ-GFP",
    "HEK293 (endogenous HiBiT-BRD4 knock-in)": "endogenous HiBiT-BRD4 knock-in",
    "HEK293 (endogenous HiBiT–Brd2)": "endogenous HiBiT–Brd2",
    "HEK293 (endogenous HiBiT–Brd3)": "endogenous HiBiT–Brd3",
    "HEK293 (endogenous HiBiT–Brd4)": "endogenous HiBiT–Brd4",
    "HEK293 (stable 3CLpro expression)": "stable 3CLpro expression",
    "HEK293 (stably transfected with α1A-AR)": "stably transfected with α1A-AR",
    "HEK293 CRISPR HiBiT BRD4 cell line": "CRISPR HiBiT BRD4",
    "HEK293 GSK-3β-HiBiT knock-in": "GSK-3β-HiBiT knock-in",
    "HEK293 Halo-VPS34 knock-in (Clone 56)": "Halo-VPS34 knock-in",
    "HEK293 HiBiT-BRD4 Cln3": "HiBiT-BRD4",
    "HEK293 LgBiT stable cells": "LgBiT",
    "HEK293 SGK3-Halo knock-in (Clone 2)": "SGK3-Halo knock-in",
    "HEK293 SNAP-EGFP": "SNAP-EGFP",
    "HEK293 cells stably transfected with Mpro": "stably transfected with Mpro",
    "HEK293 expressing BTK": "expressing BTK",
    "HEK293-HiBiT": "HiBiT",
    "HEK293-hTau overexpressed cell line": "hTau overexpressed",
    "HEK293-hTau, SH-SY5Y (p-tau induced using okadaic acid)": "hTau",
    "HEK293T (AR-GFP transfected)": "AR-GFP transfected",
    "HEK293T (GFP-HaloTag-expressing)": "GFP-HaloTag-expressing",
    "HEK293T (GFP-NCL-expressing)": "GFP-NCL-expressing",
    "HEK293T (USP39-OFP + VHL_WT-EGFP)": "USP39-OFP + VHL_WT-EGFP",
    "HEK293T (stably expressing 3CPro-Flag)": "stably expressing 3CPro-Flag",
    "HEK293T BRD4-HiBiT knock-in cells": "BRD4-HiBiT",
    "HEK293T HiBiT-TEAD1": "HiBiT-TEAD1",
    "HEK293T HiBiT-TEAD2": "HiBiT-TEAD2",
    "HEK293T HiBiT-TEAD3": "HiBiT-TEAD3",
    "HEK293T HiBiT-TEAD4": "HiBiT-TEAD4",
    "HEK293T-NSD2-HiBiT": "NSD2-HiBiT",
    "HEK293T/17 (SHP2-sfGFPxK stable line)": "SHP2-sfGFPxK stable line",
    "HEK293TBRD4‑HiBiT": "BRD4‑HiBiT",
    "HeLa (HiBiT-KAT6A)": "HiBiT-KAT6A",
    "HeLa (HiBiT-SMARCA2)": "HiBiT-SMARCA2",
    "HeLa (HiBiT-SMARCA4)": "HiBiT-SMARCA4",
    "HeLa (HttQ97-NES-eGFP)": "HttQ97-NES-eGFP",
    "HeLa (HttQ97-NLS-eGFP)": "HttQ97-NLS-eGFP",
    "HeLa (overexpressed mutant)": "overexpressed mutant",
    "HeLa, stable HA-tagged Mpro": "stable HA-tagged Mpro",
    "HeLasEH-HiBiT": "sEH-HiBiT",
    "HiBiT-BRD4 HEK293 CRISPR cells": "HiBiT-BRD4",
    "HiBiT-BromoTag-BRD4 HEK293": "HiBiT-BromoTag-BRD4",
    "HiBiT-KRASG12D GP5d": "HiBiT-KRASG12D",
    "HiBiT-KRASG12R Cal-62": "HiBiT-KRASG12R",
    "Insig-silenced HepG2 cells": "Insig-silenced",
    "Jurkat (ITKHiBiT)": "ITKHiBiT",
    "Jurkat BRD4-HiBiT cells": "BRD4-HiBiT",
    "Jurkat nLuc-RIPK1": "nLuc-RIPK1",
    "K562 HiBiT": "HiBiT",
    "K562-Cas9": "Cas9",
    "KBM7 iCas9": "iCas9",
    "KRAS G12C-expressing cell lines": "KRAS G12C-expressing",
    "KRAS G12D-HiBiT KI HEK293": "KRAS G12D-HiBiT KI",
    "KRAS G12V-HiBiT KI SW620": "KRAS G12V-HiBiT KI",
    "MCF7 (ESR1-NanoLuc reporter)": "ESR1-NanoLuc reporter",
    "MV4-11 (HiBiT-USP7)": "HiBiT-USP7",
    "MV4-11 (HiBiT-WDR5)": "HiBiT-WDR5",
    "MV4-11AURKA-HiBiT": "AURKA-HiBiT",
    "MV4-11AURKB-HiBiT": "AURKB-HiBiT",
    "MV4-11AURORA‑A‑HiBiT": "AURORA‑A‑HiBiT",
    "MV4-11AURORA‑A‑Nluc(Kless)": "AURORA‑A‑Nluc(Kless)",
    "MV4-11HiBiT-TTK": "HiBiT-TTK",
    "MV4–11 AURKAHiBiT": "AURKAHiBiT",
    "NCI-H2052 (HiBiT-TEAD1)": "HiBiT-TEAD1",
    "NCI-H2052 XIAP KO": "XIAP KO",
    "NCI-H2052 cIAP1 KO": "cIAP1 KO",
    "NCI-H226 (HiBiT-TEAD1)": "HiBiT-TEAD1",
    "NCI-H226 (HiBiT-TEAD2)": "HiBiT-TEAD2",
    "NCI-H226 (HiBiT-TEAD3)": "HiBiT-TEAD3",
    "NCI-H226 (HiBiT-TEAD4)": "HiBiT-TEAD4",
    "PC-3 (FKBP12F36V-EGFP reporter)": "FKBP12F36V-EGFP reporter",
    "PC12 cells and PC12 transfected with GFT-tau plasmid": "transfected with GFT-tau plasmid",
    "Q175 knock-in mouse cortical neurons": "Q175 knock-in",
    "RIPK2 HiBiT HEK293": "RIPK2 HiBiT",
    "SHEP MYCN-ER": "MYCN-ER",
    "STAT3-HiBiT HeLa": "STAT3-HiBiT",
    "Tau overexpressed cell lines (SH-SY5Y, N2a and PC-12)": "Tau overexpressed",
    "Tau-EGFP-overexpressed N2a cells, tau-overexpressed SH-SY5Y cells, 3×Tg-AD mouse primary neuron cells": "Tau-EGFP-overexpressed",
    "U251 (overexpressing α-synA53T)": "overexpressing α-synA53T",
    "U2OS KLHDC2 KO/rescue": "KLHDC2 KO/rescue",
    "mouse bone marrow-derived cells (BMDCs, CDK2dTAG)": "CDK2dTAG",
    "mouse bone marrow-derived cells (BMDCs, CDK5dTAG)": "CDK5dTAG",
    "mouse duodenum organoids (CDK2dTAG)": "CDK2dTAG",
    "mouse duodenum organoids (CDK5dTAG)": "CDK5dTAG",
    "p53-R175H-expressing cancer cells": "p53-R175H-expressing",
    "ΔG3BP1 + Halotag": "ΔG3BP1 + Halotag",
}


def standardize_cell_lines(df_llm: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cl_list = sorted(
        df_llm["Cell_Line"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )
    print(f"Unique Cell_Line values: {len(cl_list)}")

    cl_info: dict[str, dict] = {}
    not_found = []

    for i, cl in enumerate(cl_list):
        print(f"[{i+1}/{len(cl_list)}] {cl}", end="")
        search_name = CL_SEARCH_MAP.get(cl, cl)
        if search_name is None:
            not_found.append(cl)
            print(f"  SKIPPED (no standard cell line)")
            continue
        if search_name.startswith("CVCL_"):
            result = cellosaurus_fetch(search_name)
        else:
            result = cellosaurus_search(search_name)
        if result:
            cl_info[cl] = result
            if cl not in CL_MODIFICATION_MAP and cl not in CL_SEARCH_MAP and _ENGINEERING_HINT.search(cl):
                _note_unmapped("cell_line_modification", cl)
            sp = result.get("species", "")
            if result["name"] != cl:
                print(f"  -> {result['name']} ({result['accession']}, {sp})")
            else:
                print(f"  OK ({result['accession']}, {sp})")
        else:
            not_found.append(cl)
            _note_unmapped("cell_line", cl)
            if cl not in CL_MODIFICATION_MAP and _ENGINEERING_HINT.search(cl):
                _note_unmapped("cell_line_modification", cl)
            print(f"  NOT FOUND")
        time.sleep(0.15)

    print(f"\n--- Results ---")
    print(f"Found: {len(cl_info)}/{len(cl_list)}")
    print(f"Not found: {len(not_found)}")
    if not_found:
        print(f"\nNot found list:")
        for cl in not_found:
            print(f"  {cl}")

    def _cl_field(cl, field):
        if pd.notna(cl):
            return cl_info.get(cl, {}).get(field)
        return None

    _cl_stripped = df_llm["Cell_Line"].astype("string").str.strip()

    recommended = _cl_stripped.map(
        lambda cl: CL_STD_OVERRIDE.get(cl) or _cl_field(cl, "name")
    ).astype("string")
    df_llm["Cell_Line_STD"] = recommended.fillna(_cl_stripped)
    df_llm["Cell_Line_STD_Accession"] = _cl_stripped.map(lambda cl: _cl_field(cl, "accession"))
    df_llm["Cell_Line_Modification"] = _cl_stripped.map(
        lambda cl: CL_MODIFICATION_MAP.get(cl) if pd.notna(cl) else None
    ).astype("string")
    df_llm["Cell_Line_Species"] = _cl_stripped.map(lambda cl: _cl_field(cl, "species"))
    df_llm["_taxonomy_id"] = _cl_stripped.map(
        lambda cl: _cl_field(cl, "taxonomy_id") or HUMAN
    ).astype(int)

    n_rows = _cl_stripped.notna().sum()
    n_acc = df_llm["Cell_Line_STD_Accession"].notna().sum()
    n_sp = df_llm["Cell_Line_Species"].notna().sum()
    n_mod = df_llm["Cell_Line_Modification"].notna().sum()
    n_renamed = (recommended.notna() & (recommended != _cl_stripped)).sum()
    print(f"\nRows with a cell line: {n_rows}/{len(df_llm)}")
    print(f"Cellosaurus accession mapped: {n_acc}/{n_rows} rows")
    print(f"Cell_Line_STD differs from the extracted name: {n_renamed} rows")
    print(f"Cell_Line_STD kept as extracted (no Cellosaurus entry): {n_rows - n_acc} rows")
    print(f"Cell_Line_Modification recorded: {n_mod}/{n_rows} rows")
    print(f"Cell_Line_Species mapped: {n_sp}/{n_rows} rows")

    unknown_mods = sorted(set(CL_MODIFICATION_MAP) - set(cl_list))
    if unknown_mods and len(cl_list) > 1:
        print(f"CL_MODIFICATION_MAP entries not present in this dataset: {len(unknown_mods)}")

    df_llm = _place_after(df_llm, "Cell_Line", [
        "Cell_Line_STD", "Cell_Line_STD_Accession",
        "Cell_Line_Modification", "Cell_Line_Species",
    ])

    return df_llm, cl_info


def build_taxid_to_species(cl_info: dict) -> dict:
    taxid_to_species = {HUMAN: "Homo sapiens"}
    for info in cl_info.values():
        tid = info.get("taxonomy_id")
        sp = info.get("species")
        if tid and sp:
            taxid_to_species[tid] = sp
    return taxid_to_species


def parse_mutation(mut_str: str | None) -> tuple[str, int, str] | None:
    if not mut_str:
        return None
    m = re.match(r'^([A-Z])(\d+)([A-Z])$', mut_str)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None


def hgnc_fetch(symbol: str) -> str | None:
    try:
        r = requests.get(
            f"https://rest.genenames.org/fetch/symbol/{quote(symbol)}",
            headers={"Accept": "application/json"}, timeout=10,
        )
        if r.ok:
            docs = r.json().get("response", {}).get("docs", [])
            if docs:
                return docs[0]["symbol"]
    except:
        pass
    return None


def hgnc_search(query: str) -> str | None:
    try:
        r = requests.get(
            f"https://rest.genenames.org/search/{quote(query)}",
            headers={"Accept": "application/json"}, timeout=10,
        )
        if r.ok:
            docs = r.json().get("response", {}).get("docs", [])
            if docs:
                return docs[0]["symbol"]
    except:
        pass
    return None


def uniprot_gene_search(query: str, organism_id: int = HUMAN) -> str | None:
    url = "https://rest.uniprot.org/uniprotkb/search"
    for q in [
        f"(gene:{query}) AND (organism_id:{organism_id}) AND (reviewed:true)",
        f'(protein_name:"{query}") AND (organism_id:{organism_id}) AND (reviewed:true)',
    ]:
        try:
            r = requests.get(url, params={
                "query": q, "format": "tsv",
                "fields": "accession,gene_primary", "size": 1,
            }, headers={"User-Agent": "protac-normalize/1.0"}, timeout=10)
            if r.ok:
                lines = r.text.strip().split("\n")
                if len(lines) > 1:
                    gene = lines[1].split("\t")[1].strip()
                    if gene:
                        return gene
        except:
            pass
    return None


def find_gene_symbol(search_term: str, organism_id: int = HUMAN) -> tuple[str | None, str]:
    if organism_id == HUMAN:
        gene = hgnc_fetch(search_term)
        if gene:
            return gene, f"HGNC fetch '{search_term}'"
        time.sleep(0.1)
        gene = hgnc_search(search_term)
        if gene:
            return gene, f"HGNC search '{search_term}'"
        time.sleep(0.1)

    gene = uniprot_gene_search(search_term, organism_id)
    if gene:
        return gene, f"UniProt({organism_id}) '{search_term}'"

    return None, "NOT FOUND"


def fetch_uniprot_accession_and_seq(gene_symbol: str, organism_id: int = HUMAN) -> tuple[str | None, str | None]:
    if not gene_symbol:
        return None, None
    url = "https://rest.uniprot.org/uniprotkb/search"
    query = f"(gene_exact:{gene_symbol}) AND (organism_id:{organism_id}) AND (reviewed:true)"
    try:
        r = requests.get(url, params={
            "query": query, "format": "tsv",
            "fields": "accession,gene_primary,sequence", "size": 1,
        }, headers={"User-Agent": "protac-normalize/1.0"}, timeout=15)
        if r.ok:
            lines = r.text.strip().split("\n")
            if len(lines) > 1:
                parts = lines[1].split("\t")
                accession = parts[0].strip() if len(parts) > 0 else None
                sequence = parts[2].strip() if len(parts) > 2 else None
                return accession, sequence
    except Exception as e:
        print(f"  UniProt accession+seq error for {gene_symbol}: {e}")
    return None, None


def fetch_uniprot_record_by_accession(accession: str) -> dict | None:
    accession = str(accession).strip()
    if not accession:
        return None
    url = f"https://rest.uniprot.org/uniprotkb/{quote(accession)}.json"
    try:
        r = requests.get(url, headers={"User-Agent": "protac-normalize/1.0"}, timeout=15)
        if not r.ok:
            return None
        data = r.json()
        genes = data.get("genes") or []
        gene = None
        if genes:
            gene = (genes[0].get("geneName") or {}).get("value")
        sequence = (data.get("sequence") or {}).get("value")
        return {
            "gene": gene,
            "uniprot": data.get("primaryAccession") or accession,
            "sequence": sequence,
        }
    except Exception as e:
        print(f"  UniProt accession fetch error for {accession}: {e}")
    return None

_variation_cache: dict[str, list] = {}


def fetch_variation_features(accession: str) -> list:
    if accession in _variation_cache:
        return _variation_cache[accession]
    url = f"https://www.ebi.ac.uk/proteins/api/variation/{accession}"
    try:
        r = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
        if r.ok:
            features = r.json().get("features", [])
            _variation_cache[accession] = features
            return features
    except Exception as e:
        print(f"  Variation API error for {accession}: {e}")
    _variation_cache[accession] = []
    return []


def find_var_id(accession: str, wt_aa: str, position: int, mut_aa: str) -> str | None:
    features = fetch_variation_features(accession)
    for f in features:
        if (str(f.get("begin", "")) == str(position)
                and f.get("wildType") == wt_aa
                and f.get("alternativeSequence") == mut_aa):
            return f.get("ftId")
    return None


def clinvar_check_variant(gene: str, mutation_str: str) -> str | None:
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        r = requests.get(url, params={
            "db": "clinvar",
            "term": f"{gene}[gene] AND {mutation_str}",
            "retmode": "json",
        }, timeout=10)
        if r.ok:
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            if ids:
                return ids[0]
    except:
        pass
    return None


def apply_mutation(sequence: str, wt_aa: str, position: int, mut_aa: str) -> str | None:
    idx = position - 1
    if idx < 0 or idx >= len(sequence):
        print(f"  apply_mutation: position {position} out of range (seq length {len(sequence)})")
        return None
    if sequence[idx] != wt_aa:
        print(f"  apply_mutation: expected {wt_aa} at position {position}, found {sequence[idx]}")
        return None
    return sequence[:idx] + mut_aa + sequence[idx + 1:]

poi_alias_map = {
    "BCL-XL": "Bcl-xL",
    "BCL-xL": "Bcl-xL",
    "Bcl-xL": "Bcl-xL",
    "Bcl-xl": "Bcl-xL",
    "Brd2": "BRD2",
    "Brd3": "BRD3",
    "Brd4": "BRD4",
}

POI_LOOKUP = {
    "AXL":        {"search": "AXL",      "mutation": None},
    "BCL-2":      {"search": "BCL2",     "mutation": None},
    "BRD2":       {"search": "BRD2",     "mutation": None},
    "BRD3":       {"search": "BRD3",     "mutation": None},
    "BRD4":       {"search": "BRD4",     "mutation": None},
    "BRD4BD1":    {"search": "BRD4",     "mutation": None},
    "BRD4BD2":    {"search": "BRD4",     "mutation": None},
    "BRD7":       {"search": "BRD7",     "mutation": None},
    "BRD9":       {"search": "BRD9",     "mutation": None},
    "BTK":        {"search": "BTK",      "mutation": None},
    "CDK4":       {"search": "CDK4",     "mutation": None},
    "CDK6":       {"search": "CDK6",     "mutation": None},
    "CDK12":      {"search": "CDK12",    "mutation": None},
    "CDK13":      {"search": "CDK13",    "mutation": None},
    "CRBN":       {"search": "CRBN",     "mutation": None},
    "FER":        {"search": "FER",      "mutation": None},
    "GDI2":       {"search": "GDI2",     "mutation": None},
    "GSPT1":      {"search": "GSPT1",    "mutation": None},
    "HDAC1":      {"search": "HDAC1",    "mutation": None},
    "HDAC2":      {"search": "HDAC2",    "mutation": None},
    "HDAC3":      {"search": "HDAC3",    "mutation": None},
    "HDAC4":      {"search": "HDAC4",    "mutation": None},
    "HDAC6":      {"search": "HDAC6",    "mutation": None},
    "HDAC8":      {"search": "HDAC8",    "mutation": None},
    "HMGCR":      {"search": "HMGCR",    "mutation": None},
    "IDO1":       {"search": "IDO1",     "mutation": None},
    "IKZF1":      {"search": "IKZF1",    "mutation": None},
    "IKZF3":      {"search": "IKZF3",    "mutation": None},
    "KEAP1":      {"search": "KEAP1",    "mutation": None},
    "LRRK2":      {"search": "LRRK2",    "mutation": None},
    "MDM2":       {"search": "MDM2",     "mutation": None},
    "MERTK":      {"search": "MERTK",    "mutation": None},
    "MET":        {"search": "MET",      "mutation": None},
    "NR4A1":      {"search": "NR4A1",    "mutation": None},
    "PARP1":      {"search": "PARP1",    "mutation": None},
    "PARP2":      {"search": "PARP2",    "mutation": None},
    "PBRM1":      {"search": "PBRM1",    "mutation": None},
    "REXO4":      {"search": "REXO4",    "mutation": None},
    "RIPK2":      {"search": "RIPK2",    "mutation": None},
    "S100A4":     {"search": "S100A4",   "mutation": None},
    "SGK3":       {"search": "SGK3",     "mutation": None},
    "SLC2A3":     {"search": "SLC2A3",   "mutation": None},
    "SLC38A2":    {"search": "SLC38A2",  "mutation": None},
    "SLC9A1":     {"search": "SLC9A1",   "mutation": None},
    "SMARCA2":    {"search": "SMARCA2",  "mutation": None},
    "SMARCA4":    {"search": "SMARCA4",  "mutation": None},
    "TYRO3":      {"search": "TYRO3",    "mutation": None},
    "WDR5":       {"search": "WDR5",     "mutation": None},
    "XIAP":       {"search": "XIAP",     "mutation": None},

    "AURORA-A":                            {"search": "AURKA",    "mutation": None},
    "Androgen receptor (AR)":              {"search": "Androgen receptor",       "mutation": None},
    "Bcl-xL":                              {"search": "BCL2L1",   "mutation": None},
    "Brd4 long":                           {"search": "BRD4",     "mutation": None},
    "Brd4 short":                          {"search": "BRD4",     "mutation": None},
    "Cdc20":                               {"search": "CDC20",    "mutation": None},
    "Cyclophilin A (CypA)":                {"search": "PPIA",     "mutation": None},
    "EGFRWT":                              {"search": "EGFR",     "mutation": None},
    "ENL":                                 {"search": "MLLT1",    "mutation": None},
    "ENL (WT and mutant)":                 {"search": "MLLT1",    "mutation": None},
    "ERK1/2":                              {"search": "MAPK1",    "mutation": None},
    "FAK":                                 {"search": "PTK2",     "mutation": None},
    "FKBP12":                              {"search": "FKBP1A",   "mutation": None},
    "GSK-3β":                              {"search": "GSK3B",    "mutation": None},
    "HSP90α":                              {"search": "HSP90AA1", "mutation": None},
    "HSP90β":                              {"search": "HSP90AB1", "mutation": None},
    "IMPDH":                               {"search": "IMPDH1",   "mutation": None},
    "MIF":                                 {"search": "MIF",      "mutation": None},
    "PARP-1":                              {"search": "PARP1",    "mutation": None},
    "PDEδ":                                {"search": "PDE6D",    "mutation": None},
    "PIKfyve":                             {"search": "PIKFYVE",  "mutation": None},
    "SHP2":                                {"search": "PTPN11",   "mutation": None},
    "Squalene synthase (SQS; FDFT1)":      {"search": "FDFT1",    "mutation": None},
    "TC-PTP":                              {"search": "PTPN2",    "mutation": None},
    "Tau (P-tau S396)":                    {"search": "MAPT",     "mutation": None},
    "Tau (insoluble fraction)":            {"search": "MAPT",     "mutation": None},
    "Tau (total)":                         {"search": "MAPT",     "mutation": None},
    "tau":                                 {"search": "MAPT",     "mutation": None},
    "c-MET":                               {"search": "MET",      "mutation": None},
    "cIAP1":                               {"search": "BIRC2",    "mutation": None},
    "cIAP2":                               {"search": "BIRC3",    "mutation": None},
    "hCAII":                               {"search": "CA2",      "mutation": None},
    "p38 MAPK":                            {"search": "MAPK14",   "mutation": None},
    "p38α":                                {"search": "MAPK14",   "mutation": None},
    "p38δ":                                {"search": "MAPK13",   "mutation": None},
    "pVHL30":                              {"search": "VHL",      "mutation": None},
    "phosphorylated p38 MAPK (p-p38)":     {"search": "MAPK14",   "mutation": None},
    "α-Synuclein (αSyn)":                  {"search": "SNCA",     "mutation": None},
    "α-synuclein aggregates":              {"search": "SNCA",     "mutation": None},
    "α1A-adrenergic receptor (α1A-AR)":    {"search": "ADRA1A", "mutation": None},

    "BRAF G466V":    {"search": "BRAF", "mutation": "G466V"},
    "BRAF G469A":    {"search": "BRAF", "mutation": "G469A"},
    "BRAF V600E":    {"search": "BRAF", "mutation": "V600E"},
    "BRAF-p61V600E": {"search": "BRAF", "mutation": "V600E"},
    "BRAFV600E":     {"search": "BRAF", "mutation": "V600E"},
    "Bcr-AblT315I":  {"search": "ABL1", "mutation": "T315I"},
    "KRASG12C":      {"search": "KRAS", "mutation": "G12C"},

    "Alpha-tubulin": {"search": None,  "mutation": None},
    "BCR-ABL":       {"search": "ABL1",    "mutation": None},
    "BRD4-L":        {"search": "BRD4",    "mutation": None},
    "Beta3-tubulin": {"search": "TUBB3",   "mutation": None},
    "EML4-ALK":      {"search": "ALK",     "mutation": None},
    "HSP90":         {"search": "HSP90AA1", "mutation": None},
    "NS3":           {"search": None,      "mutation": None},
    "c-Src":         {"search": "SRC",     "mutation": None},
    "p-p38":         {"search": "MAPK14",  "mutation": None},

    "ALK G1202R":             {"search": "ALK",   "mutation": "G1202R"},
    "AR L702H and T878A":     {"search": "AR",    "mutation": None},
    "AR T878A":               {"search": "AR",    "mutation": "T878A"},
    "BCR-ABL T315I":          {"search": "ABL1",  "mutation": "T315I"},
    "BTK C481A":              {"search": "BTK",   "mutation": "C481A"},
    "BTK C481G":              {"search": "BTK",   "mutation": "C481G"},
    "BTK C481S":              {"search": "BTK",   "mutation": "C481S"},
    "BTK C481T":              {"search": "BTK",   "mutation": "C481T"},
    "BTK C481W":              {"search": "BTK",   "mutation": "C481W"},
    "EGFR Del19":             {"search": "EGFR",  "mutation": None},
    "EGFR Del19/T790M/C797S": {"search": "EGFR",  "mutation": None},
    "EGFR L858R":             {"search": "EGFR",  "mutation": "L858R"},
    "EGFR L858R/T790M":       {"search": "EGFR",  "mutation": None},
    "EML4-ALK C1156Y":        {"search": "ALK",   "mutation": "C1156Y"},
    "EML4-ALK G1202R":        {"search": "ALK",   "mutation": "G1202R"},
    "EML4-ALK L1196M":        {"search": "ALK",   "mutation": "L1196M"},
    "EML4-ALK L1196M/G1202R": {"search": "ALK",   "mutation": None},
    "JAK2 R683G":             {"search": "JAK2",  "mutation": "R683G"},
    "KRAS G12C":              {"search": "KRAS",  "mutation": "G12C"},
    "KRAS G12D":              {"search": "KRAS",  "mutation": "G12D"},
    "LRRK2 G2019S":           {"search": "LRRK2", "mutation": "G2019S"},

    "HCV NS3": {"search": None, "mutation": None},

    "Androgen receptor (f-AR)":             {"search": "Androgen receptor", "mutation": None},
    "BCL6":                                 {"search": "BCL6",    "mutation": None},
    "CK1α":                                 {"search": "CSNK1A1", "mutation": None},
    "Cyclin K (CCNK)":                      {"search": "CCNK",    "mutation": None},
    "cyclin K":                             {"search": "CCNK",    "mutation": None},
    "cyclin K (CCNK)":                      {"search": "CCNK",    "mutation": None},
    "IKZF2":                                {"search": "IKZF2",   "mutation": None},
    "Lin28":                                {"search": "LIN28A",  "mutation": None},
    "SALL4":                                {"search": "SALL4",   "mutation": None},

    "BRD4-S":                               {"search": "BRD4",    "mutation": None},
    "CDK12-CCNK":                           {"search": None,      "mutation": None},
    "CDO1":                                 {"search": "CDO1",    "mutation": None},
    "GEMIN3":                               {"search": "DDX20",   "mutation": None},
    "GSPT2":                                {"search": "GSPT2",   "mutation": None},
    "IKZF4":                                {"search": "IKZF4",   "mutation": None},
    "Lin28A":                               {"search": "LIN28A",  "mutation": None},
    "Lin28B":                               {"search": "LIN28B",  "mutation": None},
    "NEK7":                                 {"search": "NEK7",    "mutation": None},
    "PDE6D":                                {"search": "PDE6D",   "mutation": None},
    "PLZF":                                 {"search": "ZBTB16",  "mutation": None},
    "RIOK2":                                {"search": "RIOK2",   "mutation": None},
    "WEE1":                                 {"search": "WEE1",    "mutation": None},
    "WIZ":                                  {"search": "WIZ",     "mutation": None},
    "ZBTB16":                               {"search": "ZBTB16",  "mutation": None},
    "ZNF-90":                               {"search": "ZNF90",   "mutation": None},
    "dALK3":                                {"search": "ALK",     "mutation": None},
    "eGFP(-SD36)":                          {"search": None,      "mutation": None},
    "eGFP(-SD40)":                          {"search": None,      "mutation": None},

    "FLuc-S4D (Firefly luciferase fused to SALL4 degron)": {"search": "SALL4", "mutation": None},
    "eGFP (fused to SD36)":                 {"search": None, "mutation": None},
    "eGFP (fused to SD40)":                 {"search": None, "mutation": None},
    "eGFP (fused to SD40 F18Q)":            {"search": None, "mutation": None},
    "eGFP (fused to SD56)":                 {"search": None, "mutation": None},

    "BRAFG466V":                            {"search": "BRAF",   "mutation": "G466V"},
    "BRAFG469A":                            {"search": "BRAF",   "mutation": "G469A"},
    "Flag-tagged p38":                      {"search": "MAPK14", "mutation": None},
    "ALK":                                  {"search": "ALK",     "mutation": None},
    "Bcl6":                                 {"search": "BCL6",    "mutation": None},
    "c-Myc":                                {"search": "MYC",     "mutation": None},
    "CA9":                                  {"search": "CA9",     "mutation": None},
    "CDCP1":                                {"search": "CDCP1",   "mutation": None},
    "CDK9 (WEIS-tagged)":                   {"search": "CDK9",    "mutation": None},
    "Cyclin K":                             {"search": "CCNK",    "mutation": None},
    "DDX52":                                {"search": "DDX52",   "mutation": None},
    "DENV2 capsid (C) protein, 16681 C-S34L mutant": {"search": None, "mutation": None},
    "DENV2 capsid (C) protein, 16681 strain (WT)":   {"search": None, "mutation": None},
    "DENV2 capsid (C) protein, NGC strain":          {"search": None, "mutation": None},
    "eGFP":                                 {"search": None,      "mutation": None},
    "EGFP (WEIS-tagged)":                   {"search": None,      "mutation": None},
    "EGFR":                                 {"search": "EGFR",    "mutation": None},
    "FIZ1 ZF8 (EGFP fusion)":               {"search": "FIZ1",    "mutation": None},
    "G3BP2":                                {"search": "G3BP2",   "mutation": None},
    "GSPT1-GFP":                            {"search": "GSPT1",   "mutation": None},
    "HBS1L":                                {"search": "HBS1L",   "mutation": None},
    "HER2":                                 {"search": "ERBB2",   "mutation": None},
    "HPRT1 (WEIS-tagged)":                  {"search": "HPRT1",   "mutation": None},
    "HuR (ELAVL1)":                         {"search": "ELAVL1",  "mutation": None},
    "IKZF1 ZF2-GFP":                        {"search": "IKZF1",   "mutation": None},
    "IKZF1, IKZF3":                         {"search": None,      "mutation": None},
    "IMP3":                                 {"search": "IMP3",    "mutation": None},
    "Integrin β4":                           {"search": "ITGB4",   "mutation": None},
    "KDM4B":                                {"search": "KDM4B",   "mutation": None},
    "LCK":                                  {"search": "LCK",     "mutation": None},
    "LDHB":                                 {"search": "LDHB",    "mutation": None},
    "NanoLuc (WEIS-tagged)":                {"search": None,      "mutation": None},
    "NSD2":                                 {"search": "NSD2",    "mutation": None},
    "PD-1":                                 {"search": "PDCD1",   "mutation": None},
    "PD-L1":                                {"search": "CD274",   "mutation": None},
    "RBM39 (RDAdel)":                       {"search": "RBM39",   "mutation": "R150_A152del"},
    "RBM39 (wt)":                           {"search": "RBM39",   "mutation": None},
    "SACK1F":                               {"search": "SACK1F",  "mutation": None},
    "SACK1G":                               {"search": "SACK1G",  "mutation": None},
    "TRIM28 (WEIS-tagged, C-terminal)":      {"search": "TRIM28",  "mutation": None},
    "TRIM28 (WEIS-tagged, N-terminal)":      {"search": "TRIM28",  "mutation": None},
    "TROP2":                                {"search": "TACSTD2", "mutation": None},
    "USP10":                                {"search": "USP10",   "mutation": None},
    "VAV1":                                 {"search": "VAV1",    "mutation": None},
    "VCL":                                  {"search": "VCL",     "mutation": None},
    "ZBTB11":                               {"search": "ZBTB11",  "mutation": None},
    "20S proteasome subunit β5": {"search": "PSMB5", "mutation": None},
    "ABL": {"search": "ABL1", "mutation": None},
    "ACAT2": {"search": "ACAT2", "mutation": None},
    "AKR1C1/C2": {"search": None, "mutation": None},
    "AKR1C3": {"search": "AKR1C3", "mutation": None},
    "AKT": {"search": "AKT1", "mutation": None},
    "ALK (EML4-ALK)": {"search": "ALK", "mutation": None},
    "ALK F1174L": {"search": "ALK", "mutation": "F1174L"},
    "ALK2": {"search": "ACVR1", "mutation": None},
    "AR": {"search": "AR", "mutation": None},
    "AR-FL": {"search": "AR", "mutation": None},
    "AR-V7": {"search": "AR", "mutation": None},
    "ARV7": {"search": "AR", "mutation": None},
    "ARv7": {"search": "AR", "mutation": None},
    "ATR": {"search": "ATR", "mutation": None},
    "AURKA": {"search": "AURKA", "mutation": None},
    "AURKB": {"search": "AURKB", "mutation": None},
    "Active YAP": {"search": "YAP1", "mutation": None},
    "Aiolos": {"search": "IKZF3", "mutation": None},
    "Akt1": {"search": "AKT1", "mutation": None},
    "Akt2": {"search": "AKT2", "mutation": None},
    "Akt3": {"search": "AKT3", "mutation": None},
    "Androgen Receptors": {"search": "AR", "mutation": None},
    "Androgen receptor": {"search": "AR", "mutation": None},
    "Androgen receptor (AR) D891H": {"search": "AR", "mutation": "D891H"},
    "Androgen receptor (AR) F877L/T878A": {"search": "AR", "mutation": None},
    "Androgen receptor (AR) H875Y/T878A": {"search": "AR", "mutation": None},
    "Androgen receptor (AR) L702H": {"search": "AR", "mutation": "L702H"},
    "Androgen receptor (AR) M750T": {"search": "AR", "mutation": "M750T"},
    "Androgen receptor (AR) M750V": {"search": "AR", "mutation": "M750V"},
    "Androgen receptor (AR) S889G": {"search": "AR", "mutation": "S889G"},
    "Androgen receptor (AR) T878A/D891H": {"search": "AR", "mutation": None},
    "Androgen receptor (AR) T878A/L702H": {"search": "AR", "mutation": None},
    "Androgen receptor (AR) T878A/S889G": {"search": "AR", "mutation": None},
    "Androgen receptor (AR) T878S": {"search": "AR", "mutation": "T878S"},
    "Androgen receptor (AR) V716M": {"search": "AR", "mutation": "V716M"},
    "Androgen receptor (AR) W742C": {"search": "AR", "mutation": "W742C"},
    "Androgen receptor (AR) W742L": {"search": "AR", "mutation": "W742L"},
    "Androgen receptor (AR) Wild-type": {"search": "AR", "mutation": None},
    "Androgen receptor (AR) and AR-V7": {"search": "AR", "mutation": None},
    "Aurora A": {"search": "AURKA", "mutation": None},
    "Aurora kinase A": {"search": "AURKA", "mutation": None},
    "Aurora-A": {"search": "AURKA", "mutation": None},
    "BAZ2A": {"search": "BAZ2A", "mutation": None},
    "BAZ2B": {"search": "BAZ2B", "mutation": None},
    "BCL-6": {"search": "BCL6", "mutation": None},
    "BCL2": {"search": "BCL2", "mutation": None},
    "BCLXL": {"search": "BCL2L1", "mutation": None},
    "BCR-ABL (T315I)": {"search": "ABL1", "mutation": "T315I"},
    "BCR-ABLT315I": {"search": "ABL1", "mutation": "T315I"},
    "BCR–ABL": {"search": "ABL1", "mutation": None},
    "BET": {"search": "BRD4", "mutation": None},
    "BET (BRD2/3/4)": {"search": "BRD4", "mutation": None},
    "BET proteins": {"search": "BRD4", "mutation": None},
    "BET proteins (BRD2/BRD3/BRD4)": {"search": "BRD4", "mutation": None},
    "BLK": {"search": "BLK", "mutation": None},
    "BPTF": {"search": "BPTF", "mutation": None},
    "BRAF": {"search": "BRAF", "mutation": None},
    "BRAF p61 V600E": {"search": "BRAF", "mutation": "V600E"},
    "BRAF p61V600E": {"search": "BRAF", "mutation": "V600E"},
    "BRAF-G466V": {"search": "BRAF", "mutation": "G466V"},
    "BRAF-G469A": {"search": "BRAF", "mutation": "G469A"},
    "BRAF-V600D": {"search": "BRAF", "mutation": "V600D"},
    "BRAF-V600E": {"search": "BRAF", "mutation": "V600E"},
    "BRD2/3/4": {"search": "BRD4", "mutation": None},
    "BRD2/BRD3/BRD4": {"search": "BRD4", "mutation": None},
    "BRD3/BRD4": {"search": "BRD4", "mutation": None},
    "BRD4 (long isoform)": {"search": "BRD4", "mutation": None},
    "BRD4 (short isoform)": {"search": "BRD4", "mutation": None},
    "BRD4 BD1": {"search": "BRD4", "mutation": None},
    "BRD4 L": {"search": "BRD4", "mutation": None},
    "BRD4 S": {"search": "BRD4", "mutation": None},
    "BRD4 [long]": {"search": "BRD4", "mutation": None},
    "BRD4 [short]": {"search": "BRD4", "mutation": None},
    "BRD4 long": {"search": "BRD4", "mutation": None},
    "BRD4 long isoform": {"search": "BRD4", "mutation": None},
    "BRD4 short": {"search": "BRD4", "mutation": None},
    "BRD4 short isoform": {"search": "BRD4", "mutation": None},
    "BRD4-BD1 reporter": {"search": "BRD4", "mutation": None},
    "BRD4-BD1-BD2 reporter": {"search": "BRD4", "mutation": None},
    "BRD4-BD2 reporter": {"search": "BRD4", "mutation": None},
    "BRD4L94V (BromoTag mutant)": {"search": "BRD4", "mutation": None},
    "BRD4Long": {"search": "BRD4", "mutation": None},
    "BRD4Short": {"search": "BRD4", "mutation": None},
    "BTK (C481S)": {"search": "BTK", "mutation": "C481S"},
    "BTK (WT)": {"search": "BTK", "mutation": None},
    "BTK (wild-type)": {"search": "BTK", "mutation": None},
    "BTK C481S mutant": {"search": "BTK", "mutation": "C481S"},
    "BTKC481S": {"search": "BTK", "mutation": "C481S"},
    "Bcl-2": {"search": "BCL2", "mutation": None},
    "Bcl-XL": {"search": "BCL2L1", "mutation": None},
    "Bcl-w": {"search": "BCL2L2", "mutation": None},
    "Brd4 (long isoform)": {"search": "BRD4", "mutation": None},
    "Brd4 (short isoform)": {"search": "BRD4", "mutation": None},
    "Brd7": {"search": "BRD7", "mutation": None},
    "Brd9": {"search": "BRD9", "mutation": None},
    "BromoCatch-Brd4": {"search": "BRD4", "mutation": None},
    "BromoTag-BRD4": {"search": "BRD4", "mutation": None},
    "BromoTag-Brd2": {"search": "BRD2", "mutation": None},
    "C-MET": {"search": "MET", "mutation": None},
    "C481S BTK": {"search": "BTK", "mutation": "C481S"},
    "CAMK2D": {"search": "CAMK2D", "mutation": None},
    "CARM1": {"search": "CARM1", "mutation": None},
    "CBP": {"search": "CREBBP", "mutation": None},
    "CBP (CREB-binding protein)": {"search": "CREBBP", "mutation": None},
    "CBP/p300": {"search": None, "mutation": None},
    "CCR2": {"search": "CCR2", "mutation": None},
    "CCR2-HaloTag-HiBiT": {"search": "CCR2", "mutation": None},
    "CD147": {"search": "BSG", "mutation": None},
    "CDK1": {"search": "CDK1", "mutation": None},
    "CDK19": {"search": "CDK19", "mutation": None},
    "CDK2": {"search": "CDK2", "mutation": None},
    "CDK4/6": {"search": "CDK4", "mutation": None},
    "CDK5": {"search": "CDK5", "mutation": None},
    "CDK7": {"search": "CDK7", "mutation": None},
    "CDK8": {"search": "CDK8", "mutation": None},
    "CDK9": {"search": "CDK9", "mutation": None},
    "CHEK2": {"search": "CHEK2", "mutation": None},
    "CHK1": {"search": "CHEK1", "mutation": None},
    "CK1δ": {"search": "CSNK1D", "mutation": None},
    "CK2": {"search": "CSNK2A1", "mutation": None},
    "CLK1": {"search": "CLK1", "mutation": None},
    "CRAF": {"search": "RAF1", "mutation": None},
    "CREPT": {"search": "RPRD1B", "mutation": None},
    "CSF-1R": {"search": "CSF1R", "mutation": None},
    "Chk1": {"search": "CHEK1", "mutation": None},
    "ClpC1 (NTD)": {"search": None, "mutation": None},
    "ClpC1 (full-length)": {"search": None, "mutation": None},
    "ClpC1-NTD": {"search": None, "mutation": None},
    "Cofilin-2": {"search": "CFL2", "mutation": None},
    "Cyclin C": {"search": "CCNC", "mutation": None},
    "Cyclin D1": {"search": "CCND1", "mutation": None},
    "Cyclin H": {"search": "CCNH", "mutation": None},
    "DAPK1": {"search": "DAPK1", "mutation": None},
    "DDR1": {"search": "DDR1", "mutation": None},
    "DHHC3": {"search": "ZDHHC3", "mutation": None},
    "DNPH1": {"search": "DNPH1", "mutation": None},
    "DOT1L": {"search": "DOT1L", "mutation": None},
    "DYRK1A": {"search": "DYRK1A", "mutation": None},
    "DYRK1B": {"search": "DYRK1B", "mutation": None},
    "Dengue virus envelope (E) protein": {"search": None, "mutation": None},
    "EED": {"search": "EED", "mutation": None},
    "EGFP": {"search": None, "mutation": None},
    "EGFP (HaloTag fusion)": {"search": None, "mutation": None},
    "EGFP-IKZF": {"search": None, "mutation": None},
    "EGFP-SD": {"search": None, "mutation": None},
    "EGFR (L858R+T790M)": {"search": "EGFR", "mutation": None},
    "EGFR (L858R/T790M)": {"search": "EGFR", "mutation": None},
    "EGFR (del19)": {"search": "EGFR", "mutation": None},
    "EGFR Ex19del": {"search": "EGFR", "mutation": None},
    "EGFR L858R+T790M": {"search": "EGFR", "mutation": None},
    "EGFR/HER2": {"search": None, "mutation": None},
    "EGFR; PARP": {"search": None, "mutation": None},
    "EGFRC797S": {"search": "EGFR", "mutation": "C797S"},
    "EGFRDel19": {"search": "EGFR", "mutation": None},
    "EGFREx19del": {"search": "EGFR", "mutation": None},
    "EGFRL858R": {"search": "EGFR", "mutation": "L858R"},
    "EGFRL858R/T790M": {"search": "EGFR", "mutation": None},
    "EGFRdel19": {"search": "EGFR", "mutation": None},
    "EGFRdel19/T790M/C797S": {"search": "EGFR", "mutation": None},
    "EGFRe19d": {"search": "EGFR", "mutation": None},
    "EML4-ALKG1202R": {"search": "ALK", "mutation": "G1202R"},
    "EML4–ALK mutation": {"search": "ALK", "mutation": None},
    "EML4–ALKWT": {"search": "ALK", "mutation": None},
    "ENL (WT)": {"search": "MLLT1", "mutation": None},
    "ENL (mutant mENL)": {"search": "MLLT1", "mutation": None},
    "EPOP–FKBP12F36V–HiBiT (endogenous fusion)": {"search": "EPOP", "mutation": None},
    "EPOP–HaloTag–HiBiT (endogenous fusion)": {"search": "EPOP", "mutation": None},
    "ER": {"search": "ESR1", "mutation": None},
    "ER (mutants; various)": {"search": "ESR1", "mutation": None},
    "ER (wild-type)": {"search": "ESR1", "mutation": None},
    "ER-α": {"search": "ESR1", "mutation": None},
    "ERBB2": {"search": "ERBB2", "mutation": None},
    "ERBB4/HER4": {"search": "ERBB4", "mutation": None},
    "ERCC1": {"search": "ERCC1", "mutation": None},
    "ERG": {"search": "ERG", "mutation": None},
    "ERK5": {"search": "MAPK7", "mutation": None},
    "ERRα": {"search": "ESRRA", "mutation": None},
    "ERα": {"search": "ESR1", "mutation": None},
    "ERβ": {"search": "ESR2", "mutation": None},
    "ERβ (Estrogen receptor beta)": {"search": "ESR2", "mutation": None},
    "EV71 3CPro": {"search": None, "mutation": None},
    "EZH2": {"search": "EZH2", "mutation": None},
    "EpCAM": {"search": "EPCAM", "mutation": None},
    "Estrogen receptor (ER)": {"search": "ESR1", "mutation": None},
    "Estrogen receptor (ERα)": {"search": "ESR1", "mutation": None},
    "Estrogen receptor alpha (ERα)": {"search": "ESR1", "mutation": None},
    "Exon 19 del EGFR": {"search": "EGFR", "mutation": None},
    "Exon 20 in. EGFR": {"search": "EGFR", "mutation": None},
    "FAK (PTK2)": {"search": "PTK2", "mutation": None},
    "FAK (total)": {"search": "PTK2", "mutation": None},
    "FAK/PTK2": {"search": "PTK2", "mutation": None},
    "FBXO22": {"search": "FBXO22", "mutation": None},
    "FES": {"search": "FES", "mutation": None},
    "FGFR1": {"search": "FGFR1", "mutation": None},
    "FGFR2": {"search": "FGFR2", "mutation": None},
    "FKBP12F36V": {"search": "FKBP1A", "mutation": "F36V"},
    "FKBP12F36V fusion proteins": {"search": None, "mutation": None},
    "FKBP12F36V-EGFP": {"search": "FKBP1A", "mutation": "F36V"},
    "FLT-3 ITD": {"search": "FLT3", "mutation": None},
    "FLT-3 WT": {"search": "FLT3", "mutation": None},
    "FLT3": {"search": "FLT3", "mutation": None},
    "FLT3-ITD": {"search": "FLT3", "mutation": None},
    "FLT3-ITD and c-KIT": {"search": None, "mutation": None},
    "FOXM1": {"search": "FOXM1", "mutation": None},
    "FRS2α": {"search": "FRS2", "mutation": None},
    "FTO": {"search": "FTO", "mutation": None},
    "FUS": {"search": "FUS", "mutation": None},
    "Fak": {"search": "PTK2", "mutation": None},
    "Fibrillarin-FKBPF36V": {"search": "FBL", "mutation": None},
    "Fibrillarin-Halo": {"search": "FBL", "mutation": None},
    "Fibrillarin-NanoLuc": {"search": "FBL", "mutation": None},
    "FoxM1": {"search": "FOXM1", "mutation": None},
    "G3BP1": {"search": "G3BP1", "mutation": None},
    "G9a": {"search": "EHMT2", "mutation": None},
    "GCN5": {"search": "KAT2A", "mutation": None},
    "GFP reporter (dTAG-BromoTag-mAID-EGFP-NLS)": {"search": None, "mutation": None},
    "GFP-Halo-LC/A1 (BoNT/A1 light chain)": {"search": None, "mutation": None},
    "GFP-HaloTag": {"search": None, "mutation": None},
    "GFP-HaloTag fusion protein": {"search": None, "mutation": None},
    "GFP-HaloTag7": {"search": None, "mutation": None},
    "GFP-KRAS": {"search": "KRAS", "mutation": None},
    "GFP-LC/A1 (BoNT/A1 light chain) co-expressed with HaloTag-B8 (VHH intermediary)": {"search": None, "mutation": None},
    "GFP-LC/A1 (BoNT/A1 light chain) co-expressed with mFKBP-B8 (VHH intermediary)": {"search": None, "mutation": None},
    "GFP–HaloTag fusion protein": {"search": None, "mutation": None},
    "GLP": {"search": "EHMT1", "mutation": None},
    "GPX4": {"search": "GPX4", "mutation": None},
    "GSK-3": {"search": "GSK3B", "mutation": None},
    "GSK-3α": {"search": "GSK3A", "mutation": None},
    "GSK3α": {"search": "GSK3A", "mutation": None},
    "GSK3β": {"search": "GSK3B", "mutation": None},
    "GSTP": {"search": "GSTP1", "mutation": None},
    "Gli1": {"search": "GLI1", "mutation": None},
    "H-FF-N-F fusion (NanoLuc-tagged)": {"search": None, "mutation": None},
    "H-PGDS": {"search": "HPGDS", "mutation": None},
    "H2B": {"search": None, "mutation": None},
    "HBV polymerase": {"search": None, "mutation": None},
    "HBc": {"search": None, "mutation": None},
    "HCV": {"search": None, "mutation": None},
    "HCV NS3/4A protease": {"search": None, "mutation": None},
    "HDAC": {"search": "HDAC1", "mutation": None},
    "HDAC1/2/3": {"search": "HDAC1", "mutation": None},
    "HDAC7": {"search": "HDAC7", "mutation": None},
    "HDACs": {"search": "HDAC1", "mutation": None},
    "HIF-1α": {"search": "HIF1A", "mutation": None},
    "HIPK4": {"search": "HIPK4", "mutation": None},
    "HIV-1 Nef": {"search": None, "mutation": None},
    "HK2": {"search": "HK2", "mutation": None},
    "HPK1": {"search": "MAP4K1", "mutation": None},
    "HRAS": {"search": "HRAS", "mutation": None},
    "Halo-VPS34": {"search": "PIK3C3", "mutation": None},
    "Halo-zDHHC20": {"search": "ZDHHC20", "mutation": None},
    "Halo-zDHHC5": {"search": "ZDHHC5", "mutation": None},
    "HaloTag": {"search": None, "mutation": None},
    "HaloTag fusion protein": {"search": None, "mutation": None},
    "HaloTag2": {"search": None, "mutation": None},
    "HaloTag7": {"search": None, "mutation": None},
    "Halotag": {"search": None, "mutation": None},
    "Hemagglutinin (influenza A)": {"search": None, "mutation": None},
    "Hemagglutinin protein (HA)": {"search": None, "mutation": None},
    "Huntingtin (Q97 mutant; cytosolic)": {"search": "HTT", "mutation": None},
    "Huntingtin (Q97 mutant; nuclear)": {"search": "HTT", "mutation": None},
    "IDO-1": {"search": "IDO1", "mutation": None},
    "IKZF1/3": {"search": None, "mutation": None},
    "IRAK1": {"search": "IRAK1", "mutation": None},
    "IRAK3": {"search": "IRAK3", "mutation": None},
    "IRAK4": {"search": "IRAK4", "mutation": None},
    "IRE1": {"search": "ERN1", "mutation": None},
    "IRE1 (E621A,E836A)": {"search": "ERN1", "mutation": None},
    "IRE1 (H579A)": {"search": "ERN1", "mutation": "H579A"},
    "IRE1α (SpyTag-HiBiT fusion)": {"search": "ERN1", "mutation": None},
    "ITK": {"search": "ITK", "mutation": None},
    "Ikaros": {"search": "IKZF1", "mutation": None},
    "Influenza A hemagglutinin (HA)": {"search": None, "mutation": None},
    "Influenza A virus nucleoprotein (NP)": {"search": None, "mutation": None},
    "Influenza hemagglutinin (HA)": {"search": None, "mutation": None},
    "Integrins (PIP-recognized; measured by PIP-Fc)": {"search": None, "mutation": None},
    "JAK": {"search": "JAK1", "mutation": None},
    "JAK1": {"search": "JAK1", "mutation": None},
    "JAK2": {"search": "JAK2", "mutation": None},
    "JAK3": {"search": "JAK3", "mutation": None},
    "K-RasG12C": {"search": "KRAS", "mutation": "G12C"},
    "KAT2A/KAT2B": {"search": None, "mutation": None},
    "KAT6A": {"search": "KAT6A", "mutation": None},
    "KDM3A": {"search": "KDM3A", "mutation": None},
    "KDM3B": {"search": "KDM3B", "mutation": None},
    "KDM4A": {"search": "KDM4A", "mutation": None},
    "KDM4C": {"search": "KDM4C", "mutation": None},
    "KDM4D": {"search": "KDM4D", "mutation": None},
    "KRAS": {"search": "KRAS", "mutation": None},
    "KRAS (G12C)": {"search": "KRAS", "mutation": "G12C"},
    "KRAS (G12D)": {"search": "KRAS", "mutation": "G12D"},
    "KRAS (G12R)": {"search": "KRAS", "mutation": "G12R"},
    "KRAS (pan-KRAS)": {"search": "KRAS", "mutation": None},
    "KRAS G12V": {"search": "KRAS", "mutation": "G12V"},
    "KRAS(G12C)": {"search": "KRAS", "mutation": "G12C"},
    "KRAS(G12D)": {"search": "KRAS", "mutation": "G12D"},
    "KRASG12D": {"search": "KRAS", "mutation": "G12D"},
    "KRASG12R": {"search": "KRAS", "mutation": "G12R"},
    "Keap1": {"search": "KEAP1", "mutation": None},
    "L858R EGFR": {"search": "EGFR", "mutation": "L858R"},
    "LDHA and LDHB": {"search": None, "mutation": None},
    "LEF1": {"search": "LEF1", "mutation": None},
    "LIMK2": {"search": "LIMK2", "mutation": None},
    "LIN28": {"search": "LIN28A", "mutation": None},
    "LIN28A": {"search": "LIN28A", "mutation": None},
    "LRG1": {"search": "LRG1", "mutation": None},
    "LSD1": {"search": "KDM1A", "mutation": None},
    "LTK": {"search": "LTK", "mutation": None},
    "LYN": {"search": "LYN", "mutation": None},
    "MAGL": {"search": "MGLL", "mutation": None},
    "MAT1": {"search": "MNAT1", "mutation": None},
    "MCL-1": {"search": "MCL1", "mutation": None},
    "MCL1": {"search": "MCL1", "mutation": None},
    "MDMX": {"search": "MDM4", "mutation": None},
    "MEK1": {"search": "MAP2K1", "mutation": None},
    "MEK2": {"search": "MAP2K2", "mutation": None},
    "MELK": {"search": "MELK", "mutation": None},
    "MEP50": {"search": "WDR77", "mutation": None},
    "METTL14": {"search": "METTL14", "mutation": None},
    "METTL3": {"search": "METTL3", "mutation": None},
    "METex14Δ-GFP": {"search": "MET", "mutation": None},
    "MLK3": {"search": "MAP3K11", "mutation": None},
    "MLKL": {"search": "MLKL", "mutation": None},
    "MTH1": {"search": "NUDT1", "mutation": None},
    "MYCN": {"search": "MYCN", "mutation": None},
    "Macrophage migration inhibitory factor (MIF)": {"search": "MIF", "mutation": None},
    "MafG": {"search": "MAFG", "mutation": None},
    "Mcl-1": {"search": "MCL1", "mutation": None},
    "Met": {"search": "MET", "mutation": None},
    "MetAP2": {"search": "METAP2", "mutation": None},
    "Methionine aminopeptidase 2 (MetAP2)": {"search": "METAP2", "mutation": None},
    "NAMPT": {"search": "NAMPT", "mutation": None},
    "NAT10": {"search": "NAT10", "mutation": None},
    "NCL": {"search": "NCL", "mutation": None},
    "NCOA4": {"search": "NCOA4", "mutation": None},
    "NF-κB p50": {"search": "NFKB1", "mutation": None},
    "NLRP3": {"search": "NLRP3", "mutation": None},
    "NPM-ALK": {"search": "ALK", "mutation": None},
    "NRF2": {"search": "NFE2L2", "mutation": None},
    "NS3/4A protease (HCV)": {"search": None, "mutation": None},
    "NSD3": {"search": "NSD3", "mutation": None},
    "NUDT1": {"search": "NUDT1", "mutation": None},
    "NUDT5": {"search": "NUDT5", "mutation": None},
    "NaV1.7 (N-terminal dTAG chimera)": {"search": "SCN9A", "mutation": None},
    "NaV1.8 (HaloTag-HiBiT chimera)": {"search": "SCN10A", "mutation": None},
    "NaV1.8 (SD40-HiBiT chimera)": {"search": "SCN10A", "mutation": None},
    "NaV1.8 (dTAG-HiBiT chimera)": {"search": "SCN10A", "mutation": None},
    "Nrf2": {"search": "NFE2L2", "mutation": None},
    "Nucleolin": {"search": "NCL", "mutation": None},
    "Nucleolin (NCL)": {"search": "NCL", "mutation": None},
    "OGA": {"search": "OGA", "mutation": None},
    "P38α/β": {"search": "MAPK14", "mutation": None},
    "PA": {"search": None, "mutation": None},
    "PAK4": {"search": "PAK4", "mutation": None},
    "PARK7": {"search": "PARK7", "mutation": None},
    "PAX3::FOXO1": {"search": None, "mutation": None},
    "PCAF": {"search": "KAT2B", "mutation": None},
    "PCAF-GCN5": {"search": None, "mutation": None},
    "PCSK9": {"search": "PCSK9", "mutation": None},
    "PDE4": {"search": "PDE4B", "mutation": None},
    "PDE4B": {"search": "PDE4B", "mutation": None},
    "PDE4B shortforms": {"search": "PDE4B", "mutation": None},
    "PDE4D": {"search": "PDE4D", "mutation": None},
    "PDE4D (long isoforms)": {"search": "PDE4D", "mutation": None},
    "PDE4D (short isoforms)": {"search": "PDE4D", "mutation": None},
    "PDE4D shortforms": {"search": "PDE4D", "mutation": None},
    "PDK1": {"search": "PDPK1", "mutation": None},
    "PDL1": {"search": "CD274", "mutation": None},
    "PI3Kα": {"search": "PIK3CA", "mutation": None},
    "PIM1": {"search": "PIM1", "mutation": None},
    "PIN1": {"search": "PIN1", "mutation": None},
    "PKMYT1": {"search": "PKMYT1", "mutation": None},
    "PLK1": {"search": "PLK1", "mutation": None},
    "PLOD2": {"search": "PLOD2", "mutation": None},
    "PRMT3": {"search": "PRMT3", "mutation": None},
    "PRMT4": {"search": "CARM1", "mutation": None},
    "PRMT4 (CARM1)": {"search": "CARM1", "mutation": None},
    "PRMT5": {"search": "PRMT5", "mutation": None},
    "PRMT7": {"search": "PRMT7", "mutation": None},
    "PTP1B": {"search": "PTPN1", "mutation": None},
    "PTPN2": {"search": "PTPN2", "mutation": None},
    "PXR": {"search": "NR1I2", "mutation": None},
    "Pin1": {"search": "PIN1", "mutation": None},
    "Progesterone receptor (PR)": {"search": "PGR", "mutation": None},
    "RAD51": {"search": "RAD51", "mutation": None},
    "RARα": {"search": "RARA", "mutation": None},
    "Ras": {"search": None, "mutation": None},
    "RBFOX1": {"search": "RBFOX1", "mutation": None},
    "RET": {"search": "RET", "mutation": None},
    "RIPK1": {"search": "RIPK1", "mutation": None},
    "ROS1": {"search": "ROS1", "mutation": None},
    "ROS1 (G2032R)": {"search": "ROS1", "mutation": "G2032R"},
    "ROS1 (L2086F)": {"search": "ROS1", "mutation": "L2086F"},
    "S100P": {"search": "S100P", "mutation": None},
    "SARS-CoV-2 3CLpro": {"search": None, "mutation": None},
    "SARS-CoV-2 Mpro": {"search": None, "mutation": None},
    "SD-EGFP": {"search": None, "mutation": None},
    "SGK-Halo": {"search": "SGK3", "mutation": None},
    "SGK3 (HaloTag fusion)": {"search": "SGK3", "mutation": None},
    "SGK3 (HaloTag-fused)": {"search": "SGK3", "mutation": None},
    "SGK3 (HaloTag7 fusion)": {"search": "SGK3", "mutation": None},
    "SHP-2": {"search": "PTPN11", "mutation": None},
    "SIRT2": {"search": "SIRT2", "mutation": None},
    "SIRT6": {"search": "SIRT6", "mutation": None},
    "Sirt": {"search": "SIRT2", "mutation": None},
    "SLC2A3 (dTAG-tagged)": {"search": "SLC2A3", "mutation": None},
    "SLC38A2 (dTAG-tagged)": {"search": "SLC38A2", "mutation": None},
    "SNAP-EGFP": {"search": None, "mutation": None},
    "SOS1": {"search": "SOS1", "mutation": None},
    "SRC-1": {"search": "NCOA1", "mutation": None},
    "STAT3": {"search": "STAT3", "mutation": None},
    "STAT5": {"search": "STAT5A", "mutation": None},
    "STAT6": {"search": "STAT6", "mutation": None},
    "STING": {"search": "STING1", "mutation": None},
    "SUZ12": {"search": "SUZ12", "mutation": None},
    "Sclerostin": {"search": "SOST", "mutation": None},
    "Sirt2": {"search": "SIRT2", "mutation": None},
    "Squalene synthase (SQS)": {"search": "FDFT1", "mutation": None},
    "Stat3": {"search": "STAT3", "mutation": None},
    "Survivin": {"search": "BIRC5", "mutation": None},
    "TACC3": {"search": "TACC3", "mutation": None},
    "TAZ": {"search": "WWTR1", "mutation": None},
    "TBK1": {"search": "TBK1", "mutation": None},
    "TCPTP": {"search": "PTPN2", "mutation": None},
    "TEAD1": {"search": "TEAD1", "mutation": None},
    "TEAD2": {"search": "TEAD2", "mutation": None},
    "TEAD3": {"search": "TEAD3", "mutation": None},
    "TEAD4": {"search": "TEAD4", "mutation": None},
    "TNKS": {"search": "TNKS", "mutation": None},
    "TNKS1": {"search": "TNKS", "mutation": None},
    "TRIB2": {"search": "TRIB2", "mutation": None},
    "TRIM24": {"search": "TRIM24", "mutation": None},
    "TRK": {"search": "NTRK1", "mutation": None},
    "TRKA": {"search": "NTRK1", "mutation": None},
    "TRKA (WT)": {"search": "NTRK1", "mutation": None},
    "TRKA (fusion)": {"search": "NTRK1", "mutation": None},
    "TTK": {"search": "TTK", "mutation": None},
    "TYR": {"search": "TYR", "mutation": None},
    "Tau": {"search": "MAPT", "mutation": None},
    "Tau (MAPT)": {"search": "MAPT", "mutation": None},
    "Tau (P301L mutant)": {"search": "MAPT", "mutation": "P301L"},
    "Tau (phospho-Ser396)": {"search": "MAPT", "mutation": None},
    "TauP301L": {"search": "MAPT", "mutation": "P301L"},
    "TfR1": {"search": "TFRC", "mutation": None},
    "Total tau": {"search": "MAPT", "mutation": None},
    "Total tau and p-tau (S202, T205, S262, S396, and S404)": {"search": "MAPT", "mutation": None},
    "Total tau and p-tau (S396, S202, and T205)": {"search": "MAPT", "mutation": None},
    "Total tau and p-tau (S404)": {"search": "MAPT", "mutation": None},
    "Total tau and p-tau (T205)": {"search": "MAPT", "mutation": None},
    "Total tau, p-tau (S202, T205, S214, S262, S396, S404), and tau oligomers (T22)": {"search": "MAPT", "mutation": None},
    "USP39": {"search": "USP39", "mutation": None},
    "USP7": {"search": "USP7", "mutation": None},
    "VEGF165": {"search": "VEGFA", "mutation": None},
    "VEGFA": {"search": "VEGFA", "mutation": None},
    "VEGFR1": {"search": "FLT1", "mutation": None},
    "VEGFR2": {"search": "KDR", "mutation": None},
    "VEGFR3": {"search": "FLT4", "mutation": None},
    "VHL": {"search": "VHL", "mutation": None},
    "VHL (pVHL30 long isoform)": {"search": "VHL", "mutation": None},
    "VPS34": {"search": "PIK3C3", "mutation": None},
    "VPS34 (HaloTag fusion)": {"search": "PIK3C3", "mutation": None},
    "VPS34 (HaloTag-fused)": {"search": "PIK3C3", "mutation": None},
    "VPS34 (HaloTag7 fusion)": {"search": "PIK3C3", "mutation": None},
    "VRK1": {"search": "VRK1", "mutation": None},
    "Wild-type BTK": {"search": "BTK", "mutation": None},
    "Wild-type EGFR": {"search": "EGFR", "mutation": None},
    "XPF": {"search": "ERCC4", "mutation": None},
    "XPO1": {"search": "XPO1", "mutation": None},
    "YTHDC1": {"search": "YTHDC1", "mutation": None},
    "ZBP1": {"search": "ZBP1", "mutation": None},
    "androgen receptor": {"search": "AR", "mutation": None},
    "androgen receptor (AR)": {"search": "AR", "mutation": None},
    "brachyury": {"search": "TBXT", "mutation": None},
    "c-ABL": {"search": "ABL1", "mutation": None},
    "c-Abl": {"search": "ABL1", "mutation": None},
    "c-MYC": {"search": "MYC", "mutation": None},
    "c-SRC": {"search": "SRC", "mutation": None},
    "cAbl": {"search": "ABL1", "mutation": None},
    "cGAS-STING": {"search": None, "mutation": None},
    "cMET": {"search": "MET", "mutation": None},
    "cMYC": {"search": "MYC", "mutation": None},
    "eDHFR-tagged FAP CAR": {"search": None, "mutation": None},
    "eEF2K": {"search": "EEF2K", "mutation": None},
    "eIF4E": {"search": "EIF4E", "mutation": None},
    "eNAMPT": {"search": "NAMPT", "mutation": None},
    "endogenous cytoplasmic HaloTag–HiBiT fusion protein": {"search": None, "mutation": None},
    "endogenous nuclear HiBiT–HaloTag fusion protein": {"search": None, "mutation": None},
    "endogenous single-pass mitochondrial membrane HaloTag–HiBiT fusion protein": {"search": None, "mutation": None},
    "estrogen receptor": {"search": "ESR1", "mutation": None},
    "estrogen receptor (ER)": {"search": "ESR1", "mutation": None},
    "estrogen receptor alpha (ERα)": {"search": "ESR1", "mutation": None},
    "estrogen-related receptor alpha (ERR)": {"search": "ESRRA", "mutation": None},
    "hRpn13Pru": {"search": "ADRM1", "mutation": None},
    "mFKBP-LC/A1 (BoNT/A1 light chain)": {"search": None, "mutation": None},
    "mSA": {"search": None, "mutation": None},
    "nTurbo-FKBP12": {"search": "FKBP1A", "mutation": None},
    "nsp14 (SARS-CoV-2 nonstructural protein 14)": {"search": None, "mutation": None},
    "nucleolin": {"search": "NCL", "mutation": None},
    "p-FAK": {"search": "PTK2", "mutation": None},
    "p-p38 MAPK": {"search": "MAPK14", "mutation": None},
    "p-tau (Ser262)": {"search": "MAPT", "mutation": None},
    "p-tau (Ser396)": {"search": "MAPT", "mutation": None},
    "p300": {"search": "EP300", "mutation": None},
    "p300/CBP": {"search": None, "mutation": None},
    "p38-MAPK": {"search": "MAPK14", "mutation": None},
    "p38β": {"search": "MAPK11", "mutation": None},
    "p53": {"search": "TP53", "mutation": None},
    "p53-R175H": {"search": "TP53", "mutation": "R175H"},
    "pYAPS127": {"search": "YAP1", "mutation": None},
    "pan TEAD": {"search": "TEAD1", "mutation": None},
    "pan YAP": {"search": "YAP1", "mutation": None},
    "sEH": {"search": "EPHX2", "mutation": None},
    "sfBRD3": {"search": "BRD3", "mutation": None},
    "sfWDS": {"search": "WDR5", "mutation": None},
    "tauP301L": {"search": "MAPT", "mutation": "P301L"},
    "tauP301L (aggregated mutant tau)": {"search": "MAPT", "mutation": "P301L"},
    "total tau (Tau5)": {"search": "MAPT", "mutation": None},
    "tyrosinase (TYR)": {"search": "TYR", "mutation": None},
    "α-Syn": {"search": "SNCA", "mutation": None},
    "α-syn aggregates": {"search": "SNCA", "mutation": None},
    "α-synuclein": {"search": "SNCA", "mutation": None},
    "α-synuclein A53T": {"search": "SNCA", "mutation": "A53T"},
    "α1A-AR": {"search": "ADRA1A", "mutation": None},
    "β-catenin–HaloTag–HiBiT (endogenous fusion)": {"search": "CTNNB1", "mutation": None},
}


def _normalize_uniprot_accession(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    return re.split(r"[;,\s]+", s)[0].strip() or None


def get_lookup_mutation_for_target(target: str | None) -> str | None:
    if not target or pd.isna(target):
        return None
    lookup = POI_LOOKUP.get(str(target))
    if lookup is None:
        return None
    mutation = lookup.get("mutation")
    return mutation if parse_mutation(mutation) else None


def clean_and_enrich_poi_from_uniprot(df_llm: pd.DataFrame, taxid_to_species: dict) -> pd.DataFrame:
    if "Target_Uniprot" in df_llm.columns:
        if "Degradation_Target_Uniprot" not in df_llm.columns:
            df_llm.rename(columns={"Target_Uniprot": "Degradation_Target_Uniprot"}, inplace=True)
        else:
            df_llm["Degradation_Target_Uniprot"] = df_llm["Degradation_Target_Uniprot"].combine_first(
                df_llm["Target_Uniprot"]
            )
            df_llm.drop(columns=["Target_Uniprot"], inplace=True)

    df_llm["Degradation_Target_Uniprot"] = (
        df_llm["Degradation_Target_Uniprot"].map(_normalize_uniprot_accession).astype("string")
    )

    pairs = (
        df_llm[["Degradation_Target", "Degradation_Target_Uniprot"]]
        .dropna(subset=["Degradation_Target", "Degradation_Target_Uniprot"])
        .drop_duplicates()
        .values.tolist()
    )
    pairs.sort(key=lambda x: (x[0], x[1]))
    print(f"Unique (target, UniProt accession) pairs: {len(pairs)}")

    record_cache: dict[str, dict | None] = {}
    poi_info: dict[tuple[str, str], dict] = {}

    for i, (poi, accession) in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {poi}  ({accession})")
        info = {"gene": None, "uniprot": accession, "mutation_id": None, "sequence": None, "source": []}

        if accession not in record_cache:
            record_cache[accession] = fetch_uniprot_record_by_accession(accession)
            time.sleep(0.1)

        record = record_cache[accession]
        if not record:
            print(f"  UniProt: accession not found")
            poi_info[(poi, accession)] = info
            continue

        info["gene"] = record["gene"]
        info["uniprot"] = record["uniprot"]
        info["sequence"] = record["sequence"]
        info["source"].append("UniProt")
        print(f"  Gene: {info['gene'] or 'MISSING'}")
        print(f"  UniProt: {info['uniprot']}  (seq length: {len(info['sequence']) if info['sequence'] else 0})")

        mutation_str = get_lookup_mutation_for_target(poi)
        parsed = parse_mutation(mutation_str)
        if parsed:
            wt, pos, mut = parsed
            var_id = find_var_id(info["uniprot"], wt, pos, mut)
            if var_id:
                info["mutation_id"] = var_id
                info["source"].append("UniProt variation")
                print(f"  Mutation {mutation_str}: {var_id}")
            else:
                print(f"  Mutation {mutation_str}: no UniProt VAR ID found")

            if info["sequence"]:
                mutated = apply_mutation(info["sequence"], wt, pos, mut)
                if mutated:
                    info["sequence"] = mutated
                    print(f"  Sequence: applied {mutation_str} at position {pos}")

        poi_info[(poi, accession)] = info

    fallback_pairs = (
        df_llm[df_llm["Degradation_Target_Uniprot"].isna()][["Degradation_Target", "_taxonomy_id"]]
        .dropna(subset=["Degradation_Target"])
        .drop_duplicates()
        .values.tolist()
    )
    fallback_pairs.sort(key=lambda x: (x[0], x[1]))
    print(f"Unique fallback (target, organism) pairs without UniProt accession: {len(fallback_pairs)}")

    fallback_info: dict[tuple[str, int], dict] = {}

    for i, (poi, org_id) in enumerate(fallback_pairs):
        org_id = int(org_id)
        print(f"[fallback {i+1}/{len(fallback_pairs)}] {poi}  ({taxid_to_species.get(org_id, str(org_id))})")
        info = {"gene": None, "uniprot": None, "mutation_id": None, "sequence": None, "source": []}

        lookup = POI_LOOKUP.get(poi)
        if lookup is None:
            print(f"  WARNING: '{poi}' not in POI_LOOKUP, skipping")
            _note_unmapped("degradation_target", poi)
            fallback_info[(poi, org_id)] = info
            continue

        search_term = lookup["search"]
        mutation_str = lookup["mutation"]

        if search_term is None:
            print(f"  (no gene)")
            fallback_info[(poi, org_id)] = info
            continue

        gene, gene_src = find_gene_symbol(search_term, org_id)
        info["gene"] = gene
        if gene:
            src_tag = gene_src.split("(")[0].split(" ")[0]
            info["source"].append(src_tag)
            print(f"  Gene: {gene}  [{gene_src}]")
        else:
            print(f"  Gene: MISSING  [{gene_src}]")
            fallback_info[(poi, org_id)] = info
            continue

        accession, sequence = fetch_uniprot_accession_and_seq(gene, org_id)
        info["uniprot"] = accession
        info["sequence"] = sequence
        if accession:
            if "UniProt" not in info["source"]:
                info["source"].append("UniProt")
            print(f"  UniProt: {accession}  (seq length: {len(sequence) if sequence else 0})")
        else:
            print(f"  UniProt: not found for gene {gene} (organism {org_id})")
            fallback_info[(poi, org_id)] = info
            continue

        parsed = parse_mutation(mutation_str)
        if parsed:
            wt, pos, mut = parsed
            if org_id == HUMAN:
                clinvar_id = clinvar_check_variant(gene, mutation_str)
                if clinvar_id:
                    info["source"].append("ClinVar")
                    print(f"  ClinVar: variant confirmed (ID={clinvar_id})")

            var_id = find_var_id(accession, wt, pos, mut)
            if var_id:
                info["mutation_id"] = var_id
                print(f"  Mutation {mutation_str}: {var_id}")
            else:
                print(f"  Mutation {mutation_str}: no UniProt VAR ID found")

            if sequence:
                mutated = apply_mutation(sequence, wt, pos, mut)
                if mutated:
                    info["sequence"] = mutated
                    print(f"  Sequence: applied {mutation_str} at position {pos}")

        fallback_info[(poi, org_id)] = info
        time.sleep(0.1)

    def _get_field(row, field):
        accession = _normalize_uniprot_accession(row.get("Degradation_Target_Uniprot"))
        if accession:
            key = (row["Degradation_Target"], accession)
            return poi_info.get(key, {}).get(field)
        key = (row["Degradation_Target"], int(row["_taxonomy_id"]))
        return fallback_info.get(key, {}).get(field)

    df_llm["Degradation_Target_Gene"] = df_llm.apply(lambda r: _get_field(r, "gene"), axis=1)
    df_llm["Degradation_Target_Uniprot"] = df_llm.apply(lambda r: _get_field(r, "uniprot"), axis=1)
    df_llm["Degradation_Target_Uniprot_MutationID"] = df_llm.apply(lambda r: _get_field(r, "mutation_id"), axis=1)
    df_llm["Degradation_Target_Sequence"] = df_llm.apply(lambda r: _get_field(r, "sequence"), axis=1)

    df_llm.drop(columns=["POI_raw", "POI_Name"], inplace=True, errors="ignore")

    mapping_records = [
        {
            "Degradation_Target": k[0],
            "UniProt": v["uniprot"],
            "Gene": v["gene"],
            "MutationID": v["mutation_id"],
            "Seq_Length": len(v["sequence"]) if v["sequence"] else None,
            "Source": "/".join(v["source"]) if v["source"] else "",
        }
        for k, v in sorted(poi_info.items())
    ]
    mapping_records.extend([
        {
            "Degradation_Target": k[0],
            "UniProt": v["uniprot"],
            "Gene": v["gene"],
            "MutationID": v["mutation_id"],
            "Seq_Length": len(v["sequence"]) if v["sequence"] else None,
            "Source": "/".join(v["source"]) if v["source"] else "",
        }
        for k, v in sorted(fallback_info.items())
    ])
    mapping_df = pd.DataFrame(mapping_records)

    n_missing_accession = df_llm["Degradation_Target_Uniprot"].isna().sum()
    n_gene = mapping_df["Gene"].notna().sum() if not mapping_df.empty else 0
    n_mut = mapping_df["MutationID"].notna().sum() if not mapping_df.empty else 0
    print(f"\nRows without Degradation_Target_Uniprot: {n_missing_accession}/{len(df_llm)}")
    print(f"Gene symbols found: {n_gene}/{len(mapping_df)}")
    print(f"Mutation IDs found: {n_mut}/{len(mapping_df)}")

    df_llm = _place_after(df_llm, "Degradation_Target", [
        "Degradation_Target_Gene", "Degradation_Target_Uniprot",
        "Degradation_Target_Uniprot_MutationID", "Degradation_Target_Sequence",
    ])

    return df_llm


def clean_and_enrich_poi(df_llm: pd.DataFrame, taxid_to_species: dict) -> pd.DataFrame:
    df_llm["Degradation_Target"] = df_llm["Degradation_Target"].astype("string").str.strip()
    df_llm["Degradation_Target"] = df_llm["Degradation_Target"].replace(poi_alias_map)

    if "Target_Uniprot" in df_llm.columns or "Degradation_Target_Uniprot" in df_llm.columns:
        print("Using provided target UniProt accessions for POI enrichment")
        return clean_and_enrich_poi_from_uniprot(df_llm, taxid_to_species)

    pairs = (
        df_llm[["Degradation_Target", "_taxonomy_id"]]
        .dropna(subset=["Degradation_Target"])
        .drop_duplicates()
        .values.tolist()
    )
    pairs.sort(key=lambda x: (x[0], x[1]))
    print(f"Unique (target, organism) pairs: {len(pairs)}")

    poi_info: dict[tuple[str, int], dict] = {}

    for i, (poi, org_id) in enumerate(pairs):
        org_id = int(org_id)
        species = taxid_to_species.get(org_id, str(org_id))
        print(f"[{i+1}/{len(pairs)}] {poi}  ({species})")
        info = {"gene": None, "uniprot": None, "mutation_id": None, "sequence": None, "source": []}

        lookup = POI_LOOKUP.get(poi)
        if lookup is None:
            print(f"  WARNING: '{poi}' not in POI_LOOKUP, skipping")
            _note_unmapped("degradation_target", poi)
            poi_info[(poi, org_id)] = info
            continue

        search_term = lookup["search"]
        mutation_str = lookup["mutation"]

        if search_term is None:
            print(f"  (no gene)")
            poi_info[(poi, org_id)] = info
            continue

        gene, gene_src = find_gene_symbol(search_term, org_id)
        info["gene"] = gene
        if gene:
            src_tag = gene_src.split("(")[0].split(" ")[0]
            info["source"].append(src_tag)
            print(f"  Gene: {gene}  [{gene_src}]")
        else:
            print(f"  Gene: MISSING  [{gene_src}]")
            poi_info[(poi, org_id)] = info
            continue

        accession, sequence = fetch_uniprot_accession_and_seq(gene, org_id)
        info["uniprot"] = accession
        info["sequence"] = sequence
        if accession:
            if "UniProt" not in info["source"]:
                info["source"].append("UniProt")
            print(f"  UniProt: {accession}  (seq length: {len(sequence) if sequence else 0})")
        else:
            print(f"  UniProt: not found for gene {gene} (organism {org_id})")
            poi_info[(poi, org_id)] = info
            continue

        parsed = parse_mutation(mutation_str)
        if not parsed:
            poi_info[(poi, org_id)] = info
            continue

        wt, pos, mut = parsed

        if org_id == HUMAN:
            clinvar_id = clinvar_check_variant(gene, mutation_str)
            if clinvar_id:
                info["source"].append("ClinVar")
                print(f"  ClinVar: variant confirmed (ID={clinvar_id})")

        var_id = find_var_id(accession, wt, pos, mut)
        if var_id:
            info["mutation_id"] = var_id
            print(f"  Mutation {mutation_str}: {var_id}")
        else:
            print(f"  Mutation {mutation_str}: no UniProt VAR ID found")

        if sequence:
            mutated = apply_mutation(sequence, wt, pos, mut)
            if mutated:
                info["sequence"] = mutated
                print(f"  Sequence: applied {mutation_str} at position {pos}")

        poi_info[(poi, org_id)] = info
        time.sleep(0.1)

    def _get_field(row, field):
        key = (row["Degradation_Target"], int(row["_taxonomy_id"]))
        return poi_info.get(key, {}).get(field)

    df_llm["Degradation_Target_Gene"] = df_llm.apply(lambda r: _get_field(r, "gene"), axis=1)
    df_llm["Degradation_Target_Uniprot"] = df_llm.apply(lambda r: _get_field(r, "uniprot"), axis=1)
    df_llm["Degradation_Target_Uniprot_MutationID"] = df_llm.apply(lambda r: _get_field(r, "mutation_id"), axis=1)
    df_llm["Degradation_Target_Sequence"] = df_llm.apply(lambda r: _get_field(r, "sequence"), axis=1)

    df_llm.drop(columns=["POI_raw", "POI_Name"], inplace=True, errors="ignore")

    mapping_df = pd.DataFrame([
        {
            "Degradation_Target": k[0],
            "Organism": taxid_to_species.get(k[1], str(k[1])),
            "Gene": v["gene"],
            "UniProt": v["uniprot"],
            "MutationID": v["mutation_id"],
            "Seq_Length": len(v["sequence"]) if v["sequence"] else None,
            "Source": "/".join(v["source"]) if v["source"] else "",
        }
        for k, v in sorted(poi_info.items())
    ])

    n_gene = mapping_df["Gene"].notna().sum()
    n_uniprot = mapping_df["UniProt"].notna().sum()
    n_mut = mapping_df["MutationID"].notna().sum()
    n_nonhuman = (mapping_df["Organism"] != "Homo sapiens").sum()
    print(f"\nUnique (target, organism) pairs: {len(mapping_df)}")
    print(f"  Non-human pairs: {n_nonhuman}")
    print(f"Gene symbols found: {n_gene}/{len(mapping_df)}")
    print(f"UniProt accessions found: {n_uniprot}/{len(mapping_df)}")
    print(f"Mutation IDs found: {n_mut}/{len(mapping_df)}")

    df_llm = _place_after(df_llm, "Degradation_Target", [
        "Degradation_Target_Gene", "Degradation_Target_Uniprot",
        "Degradation_Target_Uniprot_MutationID", "Degradation_Target_Sequence",
    ])

    return df_llm

e3_alias_map = {
    "CEREBLON":           "CRBN",
    "Cereblon":           "CRBN",
    "cereblon":           "CRBN",
    "CRBN (cereblon)":    "CRBN",
    "CRBN (pomalidomide)":"CRBN",
    "Cereblon (CRBN)":    "CRBN",
    "Unknown":            None,
}

E3_LOOKUP = {
    "CRBN":   "CRBN",
    "VHL":    "VHL",
    "IAP":    "BIRC2",
    "cIAP1":  "BIRC2",
    "MDM2":   "MDM2",
    "DCAF1":  "DCAF1",
    "HRD1":   "SYVN1",
    "KLHL20": "KLHL20",

    "CRL2KLHDC3":                  "KLHDC3",
    "DCAF16":                      "DCAF16",
    "DCAF16 (CRL4DCAF16)":         "DCAF16",
    "DDB1 (CRL4-RBX1 E3 ligase)":  "DDB1",
    "DDB1 (CRL4B)":                "DDB1",
    "DDB1-CUL4-RBX1":              "DDB1",
    "DDB1–CUL4–RBX1":              "DDB1",
    "RNF126":                      "RNF126",
    "SIAH1":                       "SIAH1",

    "FEM1B":    "FEM1B",
    "Keap1":    "KEAP1",
    "XIAP":     "XIAP",
    "CI-M6PR":                         "IGF2R",
    "CRL4B (CUL4B–RBX1–DDB1)":         "DDB1",
    "CRL4B-WDR18":                     "WDR18",
    "CRL4CRBN":                        "CRBN",
    "CXCR7":                           "ACKR3",
    "DCAF11":                          "DCAF11",
    "DCAF15":                          "DCAF15",
    "DCAF16 and FBXO22":               None,
    "DDB1":                            "DDB1",
    "DDB1 (CUL4–RBX1–DDB1)":          "DDB1",
    "FBXO22":                          "FBXO22",
    "Folate receptor (FR1/FR2)":       None,
    "IL-2R (CD25/IL-2RB/IL-2RG)":      None,
    "KLHDC3":                          "KLHDC3",
    "midnolin":                        "MIDN",
    "STUB1":                           "STUB1",
    "YPEL5–CTLH E3 ligase":             "YPEL5",
    "-": None,
    "20S proteasome (PSMA6)": "PSMA6",
    "ASGPR": "ASGR1",
    "Bardoxolone": "KEAP1",
    "CBL": "CBL",
    "CBL, CBL-B": None,
    "CBL; CBLB": None,
    "CRBN (Pomalidomide)": "CRBN",
    "CRBN (Thalidomide)": "CRBN",
    "CRBN (lenalidomide analog)": "CRBN",
    "CRBN (lenalidomide)": "CRBN",
    "CRBN (thalidomide)": "CRBN",
    "CRBN and VHL": None,
    "CRBN, VHL": None,
    "CRBN; VHL": None,
    "CRL4DCAF11": "DCAF11",
    "ClpC1": None,
    "ClpC1–P1P2 protease": None,
    "ClpP": None,
    "FBXO28": "FBXO28",
    "FBXO31": "FBXO31",
    "GID4 (CTLH E3 ligase)": "GID4",
    "GPC3": "GPC3",
    "HSP70 (adamantyl)": "HSPA1A",
    "HSP90": "HSP90AA1",
    "HSP90 complex": "HSP90AA1",
    "HaloTag-VHL": "VHL",
    "Hsc70 (HSP70)": "HSPA8",
    "HyT": None,
    "HyT (no E3 ligase)": None,
    "IAP (cIAP1)": "BIRC2",
    "IpaH9.8": None,
    "KEAP1": "KEAP1",
    "KLHDC2": "KLHDC2",
    "KLHL20 (CUL3)": "KLHL20",
    "KLHL41": "KLHL41",
    "Keap1 (CUL3–Rbx1)": "KEAP1",
    "L3MBTL3": "L3MBTL3",
    "LC3": "MAP1LC3B",
    "LC3 (autophagy-tethering)": "MAP1LC3B",
    "LC3B": "MAP1LC3B",
    "LTR-independent; clathrin-mediated endocytosis via NPGY sorting signal": None,
    "MAGEA11": "MAGEA11",
    "MDM2 (AS1411-NCL)": "MDM2",
    "MDM2 (Nutlin-3)": "MDM2",
    "MDM2 (idasanutlin)": "MDM2",
    "MDM2 (via NCL bridge)": "MDM2",
    "N-degron (UBR E3 ligases)": None,
    "N-end rule (UBR E3 ligases)": None,
    "None (E3-independent)": None,
    "None (inactive enantiomer; no E3 recruitment)": None,
    "Not specified": None,
    "PSMA": "FOLH1",
    "PSMD2": "PSMD2",
    "PSMD2 (Rpn1)": "PSMD2",
    "Pomalidomide": "CRBN",
    "RNF114": "RNF114",
    "RNF149": "RNF149",
    "RNF152": "RNF152",
    "RNF4": "RNF4",
    "RNF43": "RNF43",
    "SCF β-TRCP": "BTRC",
    "SCF(FBXO22)": "FBXO22",
    "SKP1": "SKP1",
    "SKP2": "SKP2",
    "SPOP": "SPOP",
    "SQSTM1/p62": "SQSTM1",
    "TFRC and HLA-ABC": None,
    "Thalidomide": "CRBN",
    "UBE2B": "UBE2B",
    "UBE2D1": "UBE2D1",
    "UBR": None,
    "UBR E3 ligase (N-end rule)": None,
    "UBR E3 ligases (UBR1/UBR2/UBR4/UBR5)": None,
    "UBR1": "UBR1",
    "UBR1/UBR2": None,
    "UBR1/UBR2/UBR4": None,
    "UHRF1": "UHRF1",
    "VHL (9)": "VHL",
    "VHL (CRL2VHL)": "VHL",
    "VHL (VH032)": "VHL",
    "VHL-1": "VHL",
    "VHL–CRBN (heterodimeric)": None,
    "ZYG11B": "ZYG11B",
    "ZYG11B/ZER1 (CRL2)": None,
    "cIAP": "BIRC2",
    "cIAP1, XIAP": None,
    "cereblon (CRBN)": "CRBN",
    "cereblon (pomalidomide)": "CRBN",
    "nondisclosed": None,
    "none": None,
    "not applicable (hydrophobic tagging; chaperone-mediated UPS)": None,
    "not specified": None,
    "p62/SQSTM1": "SQSTM1",
    "p62/SQSTM1 (ZZ domain autophagy receptor)": "SQSTM1",
    "pomalidomide": "CRBN",
    "thalidomide": "CRBN",
    "undisclosed": None,
    "unknown": None,
    "unspecified": None,
}


def clean_and_enrich_e3(df_llm: pd.DataFrame, taxid_to_species: dict) -> pd.DataFrame:
    df_llm["Recruiter"] = df_llm["Recruiter"].astype("string").str.strip()
    df_llm["Recruiter"] = df_llm["Recruiter"].replace(e3_alias_map)

    e3_after = sorted(df_llm["Recruiter"].dropna().unique())
    print(f"Unique Recruiter values: {len(e3_after)}")

    e3_pairs = (
        df_llm[["Recruiter", "_taxonomy_id"]]
        .dropna(subset=["Recruiter"])
        .drop_duplicates()
        .values.tolist()
    )
    e3_pairs.sort(key=lambda x: (x[0], x[1]))
    print(f"Unique (recruiter, organism) pairs: {len(e3_pairs)}")

    e3_info: dict[tuple[str, int], dict] = {}

    for i, (e3, org_id) in enumerate(e3_pairs):
        org_id = int(org_id)
        species = taxid_to_species.get(org_id, str(org_id))
        print(f"[{i+1}/{len(e3_pairs)}] {e3}  ({species})")
        info = {"gene": None, "uniprot": None, "sequence": None, "source": []}

        if e3 not in E3_LOOKUP:
            print(f"  WARNING: '{e3}' not in E3_LOOKUP, skipping")
            _note_unmapped("recruiter", e3)
            e3_info[(e3, org_id)] = info
            continue

        search_term = E3_LOOKUP[e3]
        if search_term is None:
            print(f"  SKIPPED (no standard recruiter gene)")
            e3_info[(e3, org_id)] = info
            continue

        gene, gene_src = find_gene_symbol(search_term, org_id)
        info["gene"] = gene
        if gene:
            src_tag = gene_src.split("(")[0].split(" ")[0]
            info["source"].append(src_tag)
            print(f"  Gene: {gene}  [{gene_src}]")
        else:
            print(f"  Gene: MISSING  [{gene_src}]")
            e3_info[(e3, org_id)] = info
            continue

        accession, sequence = fetch_uniprot_accession_and_seq(gene, org_id)
        info["uniprot"] = accession
        info["sequence"] = sequence
        if accession:
            if "UniProt" not in info["source"]:
                info["source"].append("UniProt")
            print(f"  UniProt: {accession}  (seq length: {len(sequence) if sequence else 0})")
        else:
            print(f"  UniProt: not found for gene {gene} (organism {org_id})")

        e3_info[(e3, org_id)] = info
        time.sleep(0.1)

    def _get_e3_field(row, field):
        key = (row["Recruiter"], int(row["_taxonomy_id"]))
        return e3_info.get(key, {}).get(field)

    df_llm["Recruiter_Gene"] = df_llm.apply(lambda r: _get_e3_field(r, "gene"), axis=1)
    df_llm["Recruiter_Uniprot"] = df_llm.apply(lambda r: _get_e3_field(r, "uniprot"), axis=1)
    df_llm["Recruiter_Sequence"] = df_llm.apply(lambda r: _get_e3_field(r, "sequence"), axis=1)

    df_llm.drop(columns=["_taxonomy_id"], inplace=True)

    e3_mapping_df = pd.DataFrame([
        {
            "Recruiter": k[0],
            "Organism": taxid_to_species.get(k[1], str(k[1])),
            "Gene": v["gene"],
            "UniProt": v["uniprot"],
            "Seq_Length": len(v["sequence"]) if v["sequence"] else None,
            "Source": "/".join(v["source"]) if v["source"] else "",
        }
        for k, v in sorted(e3_info.items())
    ])

    n_gene = e3_mapping_df["Gene"].notna().sum()
    n_uniprot = e3_mapping_df["UniProt"].notna().sum()
    n_nonhuman = (e3_mapping_df["Organism"] != "Homo sapiens").sum()
    print(f"\nUnique (recruiter, organism) pairs: {len(e3_mapping_df)}")
    print(f"  Non-human pairs: {n_nonhuman}")
    print(f"Gene symbols found: {n_gene}/{len(e3_mapping_df)}")
    print(f"UniProt accessions found: {n_uniprot}/{len(e3_mapping_df)}")

    df_llm = _place_after(df_llm, "Recruiter", [
        "Recruiter_Gene", "Recruiter_Uniprot", "Recruiter_Sequence",
    ])

    return df_llm

_RAW_COLS = ["DC50_raw", "DC50_h_raw", "Dmax_raw", "Dmax_h_raw", "Dmax_conc_raw"]
_INSERT_AFTER = {
    "DC50_h": [
        "DC50_operator", "DC50_range_min", "DC50_range_max", "DC50_parse_status",
    ],
    "Dmax_h": [
        "Dmax_operator", "Dmax_range_min", "Dmax_range_max", "Dmax_parse_status",
    ],
    "Dmax_conc": [
        "Dmax_conc_operator", "Dmax_conc_range_min",
        "Dmax_conc_range_max", "Dmax_conc_units", "Dmax_conc_parse_status",
    ],
}


def reorder_standardized_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in _RAW_COLS if c in df.columns])
    moved = {c for cols in _INSERT_AFTER.values() for c in cols}
    base_cols = [c for c in df.columns if c not in moved]
    new_cols = []
    for col in base_cols:
        new_cols.append(col)
        for follower in _INSERT_AFTER.get(col, []):
            if follower in df.columns:
                new_cols.append(follower)
    for col in df.columns:
        if col not in new_cols:
            new_cols.append(col)
    return df[new_cols]

_MONTH_TO_NUMBER = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_DASH_TRANSLATION = str.maketrans({char: "-" for char in "‐‑‒–—―−﹣－"})
_DC50_NUMBER = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
_DC50_UNIT = r"(?:pM|nM|[uµμ]M|pmol/L|nmol/L|[uµμ]mol/L)"
_TILDE_CHARS = "∼≈≃≅～"

_NO_VALUE = re.compile(
    r"n/?a|nan|none|n\.?\s*d\.?o?|not\s*determined|not\s*detected|not\s*tested|n\.?\s*t\.?",
    re.IGNORECASE,
)
_ORDER_OF_MAGNITUDE = re.compile(
    r"(?:low|high|sub|single[-\s]*digit)?\s*"
    r"(?:pico|nano|micro|milli)(?:molar|mole)\s*"
    r"(?:range|level|concentrations?|potency)?",
    re.IGNORECASE,
)
_EMPTY = {"value": None, "operator": None, "range_min": None, "range_max": None}
_DMAX_COMPLETE = re.compile(
    r"(?:almost|near(?:ly)?|close\s*to)?[-\s]*complete(?:ly)?"
    r"(?:\s*(?:depletion|degradation|loss|knockdown))?",
    re.IGNORECASE,
)
_NON_MOLAR_DOSE = re.compile(r"/\s*well|\bmg\s*/?\s*kg\b|g\s*/\s*m?[lL]\b|\bIU\b", re.IGNORECASE)
_HOUR_UNIT_FACTOR = {
    "": 1.0, "h": 1.0, "hr": 1.0, "hrs": 1.0, "hour": 1.0, "hours": 1.0,
    "min": 1 / 60, "mins": 1 / 60, "minute": 1 / 60, "minutes": 1 / 60,
    "d": 24.0, "day": 24.0, "days": 24.0,
    "week": 168.0, "weeks": 168.0, "wk": 168.0,
}


def _normalize_activity_text(raw) -> str:
    s = str(raw).replace(" ", " ").strip()
    for char in _TILDE_CHARS:
        s = s.replace(char, "~")
    s = s.replace("≥", ">=").replace("≤", "<=").replace("×", "x").replace("✕", "x")
    s = s.translate(_DASH_TRANSLATION)
    return re.sub(r"\s{2,}", " ", s).strip()


def _strip_trailing_note(s: str) -> str:
    return re.sub(r"\s*\([^()]*\)\s*$", "", s).strip()


def _is_no_value(s: str) -> bool:
    return bool(_NO_VALUE.fullmatch(s.strip("() .")))


def _recognised(field: str, raw, result):
    _note_unparsed(field, raw, recognised=True)
    return result


def _unparsed(field: str, raw, result):
    _note_unparsed(field, raw)
    print(f"  WARNING: could not parse {field} value: {raw!r}")
    return result


def _activity_parse_status(field: str, raw) -> str:
    """Return the row-level disposition recorded by an activity parser."""
    if pd.isna(raw):
        return "missing"
    normalized = _normalize_activity_text(raw)
    if not normalized or _is_no_value(normalized):
        return "missing"
    if field == "Dmax" and (
        normalized == "-" or re.fullmatch(r"-\s*[0-9.]+", normalized)
    ):
        return "missing"
    return UNPARSED.get(field, {}).get(str(raw), "parsed")


def _dc50_result(
    *,
    value=None,
    operator=None,
    range_min=None,
    range_max=None,
    units=None,
) -> dict:
    return {
        "value": value,
        "operator": operator,
        "range_min": range_min,
        "range_max": range_max,
        "units": units,
    }


def _dc50_unit_factor_to_nm(unit: str | None) -> float | None:
    if unit is None:
        return None
    normalized = unit.strip().replace("µ", "u").replace("μ", "u").lower()
    return {
        "pm": 1e-3,
        "pmol/l": 1e-3,
        "nm": 1.0,
        "nmol/l": 1.0,
        "um": 1e3,
        "umol/l": 1e3,
    }.get(normalized)


def _excel_date_range_bounds(s: str) -> tuple[float, float] | None:
    m = re.match(r"^([0-9]{1,2})-([A-Za-z]{3})$", s.strip())
    if not m:
        return None
    hi = float(m.group(1))
    lo = _MONTH_TO_NUMBER.get(m.group(2).lower())
    if lo is None:
        return None
    return float(lo), hi


def parse_dc50(raw: str) -> dict:
    if pd.isna(raw):
        return _dc50_result()

    s = _normalize_activity_text(raw)
    if not s:
        return _dc50_result()
    if _is_no_value(s):
        return _dc50_result()

    s = s.replace(",", "")
    s = re.sub(r"^DC_?50\s*[:=]?\s*", "", s, flags=re.IGNORECASE).strip()
    if _ORDER_OF_MAGNITUDE.fullmatch(s):
        return _recognised("DC50", raw, _dc50_result())

    excel_date_range = _excel_date_range_bounds(s)
    if excel_date_range:
        lo, hi = excel_date_range
        return _dc50_result(range_min=lo, range_max=hi)

    # BioPROTAC dose-response experiments may express DC50 as the percentage of
    # the transfected plasmid load rather than as a molar concentration.  Keep
    # the numeric value here; standardize_dc50 preserves the more descriptive
    # unit from the companion DC50_units column.
    m = re.fullmatch(rf"({_DC50_NUMBER})\s*%", s)
    if m:
        return _dc50_result(value=float(m.group(1)))

    m = re.fullmatch(
        rf"(<=|>=|<|>|~)?\s*({_DC50_NUMBER})\s*({_DC50_UNIT})?\s*(?:-|~)\s*"
        rf"({_DC50_NUMBER})\s*({_DC50_UNIT})?",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        operator, lo, lo_unit, hi, hi_unit = m.groups()
        operator = None if operator == "~" else operator
        lo_value, hi_value = float(lo), float(hi)
        if lo_unit or hi_unit:
            lo_factor = _dc50_unit_factor_to_nm(lo_unit or hi_unit)
            hi_factor = _dc50_unit_factor_to_nm(hi_unit or lo_unit)
            if lo_factor is not None and hi_factor is not None:
                return _dc50_result(
                    operator=operator,
                    range_min=lo_value * lo_factor,
                    range_max=hi_value * hi_factor,
                    units="nM",
                )
        return _dc50_result(operator=operator, range_min=lo_value, range_max=hi_value)

    m = re.fullmatch(
        rf"(<=|>=|<|>|~)?\s*({_DC50_NUMBER})\s*({_DC50_UNIT})?",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        operator, number, unit = m.groups()
        value = float(number)
        if unit:
            factor = _dc50_unit_factor_to_nm(unit)
            if factor is not None:
                return _dc50_result(value=value * factor, operator=operator, units="nM")
        return _dc50_result(value=value, operator=operator)

    m = re.fullmatch(
        rf"({_DC50_NUMBER})\s*±\s*({_DC50_NUMBER})\s*({_DC50_UNIT})?",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        center, error, unit = float(m.group(1)), float(m.group(2)), m.group(3)
        factor = _dc50_unit_factor_to_nm(unit) if unit else 1.0
        units = "nM" if unit else None
        return _dc50_result(
            value=center * factor,
            range_min=(center - error) * factor,
            range_max=(center + error) * factor,
            units=units,
        )

    m = re.fullmatch(
        rf"({_DC50_NUMBER})\s*x\s*10\s*\^?\s*(-?[0-9]+)\s*({_DC50_UNIT})?",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        value = round(float(m.group(1)) * 10 ** int(m.group(2)), 10)
        factor = _dc50_unit_factor_to_nm(m.group(3)) if m.group(3) else None
        if factor is not None:
            return _dc50_result(value=value * factor, units="nM")
        return _dc50_result(value=value)

    m = re.fullmatch(
        rf"[<>]=?\s*({_DC50_NUMBER})\s+and\s+[<>]=?\s*({_DC50_NUMBER})\s*({_DC50_UNIT})?",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        lo, hi = sorted((float(m.group(1)), float(m.group(2))))
        factor = _dc50_unit_factor_to_nm(m.group(3)) if m.group(3) else None
        if factor is not None:
            return _dc50_result(range_min=lo * factor, range_max=hi * factor, units="nM")
        return _dc50_result(range_min=lo, range_max=hi)

    m = re.fullmatch(
        rf"(under|below|less\s+than|up\s+to|at\s+most|more\s+than|greater\s+than|over|"
        rf"at\s+least|about|around|approximately|roughly|nearly|close\s+to)\s*"
        rf"({_DC50_NUMBER})\s*({_DC50_UNIT})?",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        word = re.sub(r"\s+", " ", m.group(1).lower())
        operator = {
            "under": "<", "below": "<", "less than": "<",
            "up to": "<=", "at most": "<=",
            "more than": ">", "greater than": ">", "over": ">",
            "at least": ">=",
        }.get(word, "~")
        value = float(m.group(2))
        factor = _dc50_unit_factor_to_nm(m.group(3)) if m.group(3) else None
        if factor is not None:
            return _dc50_result(value=value * factor, operator=operator, units="nM")
        return _dc50_result(value=value, operator=operator)

    return _unparsed("DC50", raw, _dc50_result())


def _as_hours(amount: float, unit: str) -> int | float | None:
    factor = _HOUR_UNIT_FACTOR.get(unit.strip().lower().rstrip("."))
    if factor is None:
        return None
    value = amount * factor
    return int(value) if float(value).is_integer() else round(value, 3)


def _parse_hours(field: str, raw) -> int | float | None:
    if pd.isna(raw):
        return None

    s = _normalize_activity_text(raw)
    if not s:
        return None
    if _is_no_value(s):
        return None
    if s.lower() == "overnight":
        return 16

    s = _strip_trailing_note(s)
    s = re.sub(
        r"^(?:>=|<=|>|<|~|within|up\s+to|after|about|around|approximately)\s*",
        "", s, flags=re.IGNORECASE,
    ).strip()

    excel_date_range = _excel_date_range_bounds(s)
    if excel_date_range:
        lo, hi = excel_date_range
        return _as_hours((lo + hi) / 2, "h")

    m = re.fullmatch(r"([0-9.]+)\s*-\s*([0-9.]+)\s*([A-Za-z.]*)", s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        hours = _as_hours((lo + hi) / 2, m.group(3))
        if hours is not None:
            return hours
        return _unparsed(field, raw, None)

    m = re.fullmatch(r"([0-9.]+)\s*([A-Za-z.]*)", s)
    if m:
        hours = _as_hours(float(m.group(1)), m.group(2))
        if hours is not None:
            return hours

    return _unparsed(field, raw, None)


def parse_dc50_h(raw) -> int | float | None:
    return _parse_hours("DC50_h", raw)


def _normalize_hour(v):
    if v is None or pd.isna(v):
        return pd.NA
    val = float(v)
    return int(val) if val.is_integer() else val


def _normalize_numeric(v):
    if v is None or pd.isna(v):
        return pd.NA
    val = float(v)
    return int(val) if val.is_integer() else val


def _normalize_numeric_series(series: pd.Series) -> pd.Series:
    normalized = pd.Series(
        [_normalize_numeric(v) for v in series],
        index=series.index,
        dtype=object,
    )
    has_fraction = normalized.dropna().apply(lambda v: not float(v).is_integer()).any()
    return normalized if has_fraction else normalized.astype("Int64")


def standardize_dc50(df_llm: pd.DataFrame, input_csv: str) -> pd.DataFrame:
    dc50_raw = pd.read_csv(input_csv, usecols=["DC50"])["DC50"]
    parsed = dc50_raw.apply(parse_dc50).apply(pd.Series)

    df_llm["DC50"] = _normalize_numeric_series(parsed["value"])
    df_llm["DC50_operator"] = parsed["operator"]
    df_llm["DC50_range_min"] = _normalize_numeric_series(parsed["range_min"])
    df_llm["DC50_range_max"] = _normalize_numeric_series(parsed["range_max"])
    df_llm["DC50_parse_status"] = dc50_raw.map(
        lambda raw: _activity_parse_status("DC50", raw)
    )

    extracted_units = (
        df_llm["DC50_units"]
        .astype("string")
        .str.strip()
        .replace({"nm": "nM", "nmol/L": "nM", "μM": "µM", "uM": "µM"})
    )

    df_llm["DC50_units"] = parsed["units"].astype("string").fillna(extracted_units)

    dc50_h_raw = pd.read_csv(input_csv, usecols=["DC50_h"], dtype=str)["DC50_h"]
    dc50_h_parsed = dc50_h_raw.apply(parse_dc50_h)

    dc50_h_normalized = dc50_h_parsed.map(_normalize_hour)
    has_fraction = dc50_h_normalized.dropna().apply(lambda v: not float(v).is_integer()).any()
    if has_fraction:
        df_llm["DC50_h"] = dc50_h_normalized.astype(object)
    else:
        df_llm["DC50_h"] = dc50_h_normalized.astype("Int64")

    df_llm = reorder_standardized_columns(df_llm)
    return df_llm


def parse_dmax(raw) -> dict:
    if pd.isna(raw):
        return _EMPTY.copy()

    s = _normalize_activity_text(raw)
    if not s or s == "-":
        return _EMPTY.copy()
    if _is_no_value(s):
        return _EMPTY.copy()

    if _DMAX_COMPLETE.fullmatch(s):
        return {"value": 100.0, "operator": "~", "range_min": None, "range_max": None}

    s = re.sub(r"\s*(?:%|percent)", "", s, flags=re.IGNORECASE)
    s = re.sub(
        r"\s*(?:degradation|depletion|reduction|loss|degraded|reduced|knockdown)\s*$",
        "", s, flags=re.IGNORECASE,
    ).strip()

    if re.fullmatch(r"-\s*[0-9.]+", s):
        return _EMPTY.copy()

    m = re.fullmatch(
        r"(<=|>=|<|>|~|more\s*than|greater\s*than|over|above|at\s*least|"
        r"less\s*than|under|below|up\s*to|at\s*most|"
        r"close\s*to|about|around|approximately|roughly|near(?:ly)?|approaching)?\s*"
        r"([0-9.]+)(?:\s*-\s*([0-9.]+))?",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        word = re.sub(r"\s+", "", (m.group(1) or "")).lower()
        operator = {
            "": None, "<": "<", "<=": "<=", ">": ">", ">=": ">=", "~": "~",
            "morethan": ">", "greaterthan": ">", "over": ">", "above": ">", "atleast": ">=",
            "lessthan": "<", "under": "<", "below": "<", "upto": "<=", "atmost": "<=",
        }.get(word, "~")
        lo = float(m.group(2))
        if m.group(3) is not None:
            hi = float(m.group(3))
            return {"value": None, "operator": operator, "range_min": lo, "range_max": hi}
        return {"value": lo, "operator": operator, "range_min": None, "range_max": None}

    m = re.fullmatch(r"([0-9.]+)\s*±\s*([0-9.]+)", s)
    if m:
        center, err = float(m.group(1)), float(m.group(2))
        return {"value": center, "operator": None, "range_min": center - err, "range_max": center + err}

    return _unparsed("Dmax", raw, _EMPTY.copy())


def parse_dmax_h(raw) -> int | float | None:
    return _parse_hours("Dmax_h", raw)


def standardize_dmax(df_llm: pd.DataFrame, input_csv: str) -> pd.DataFrame:
    dmax_raw = pd.read_csv(input_csv, usecols=["Dmax"])["Dmax"]
    parsed = dmax_raw.apply(parse_dmax).apply(pd.Series)

    df_llm["Dmax"] = _normalize_numeric_series(parsed["value"])
    df_llm["Dmax_operator"] = parsed["operator"]
    df_llm["Dmax_range_min"] = _normalize_numeric_series(parsed["range_min"])
    df_llm["Dmax_range_max"] = _normalize_numeric_series(parsed["range_max"])
    df_llm["Dmax_parse_status"] = dmax_raw.map(
        lambda raw: _activity_parse_status("Dmax", raw)
    )

    dmax_h_raw = pd.read_csv(input_csv, usecols=["Dmax_h"], dtype=str)["Dmax_h"]
    dmax_h_parsed = dmax_h_raw.apply(parse_dmax_h)

    dmax_h_normalized = dmax_h_parsed.map(_normalize_hour)
    has_fraction = dmax_h_normalized.dropna().apply(lambda v: not float(v).is_integer()).any()
    if has_fraction:
        df_llm["Dmax_h"] = dmax_h_normalized.astype(object)
    else:
        df_llm["Dmax_h"] = dmax_h_normalized.astype("Int64")

    df_llm = reorder_standardized_columns(df_llm)
    return df_llm

def _conc_as_nM(amount: float, unit: str) -> int | float | None:
    u = unit.strip().lower().replace("µ", "μ")
    if u in {"nm", "nmol/l"}:
        factor = 1.0
    elif u in {"μm", "um", "μmol/l", "umol/l"}:
        factor = 1000.0
    elif u in {"pm", "pmol/l"}:
        factor = 1e-3
    elif u in {"mm", "mmol/l"}:
        factor = 1e6
    else:
        return None
    val = amount * factor
    return int(val) if float(val).is_integer() else float(val)


def parse_dmax_conc(raw) -> dict:
    if pd.isna(raw):
        return _EMPTY.copy()

    s = _normalize_activity_text(raw)
    if not s or _is_no_value(s):
        return _EMPTY.copy()

    if _NON_MOLAR_DOSE.search(s):
        return _recognised("Dmax_conc", raw, _EMPTY.copy())
    if "+" in s:
        return _recognised("Dmax_conc", raw, _EMPTY.copy())
    if _ORDER_OF_MAGNITUDE.fullmatch(s):
        return _recognised("Dmax_conc", raw, _EMPTY.copy())

    s = _strip_trailing_note(s)

    m = re.fullmatch(r"([0-9.]+)\s*(\S+)\s+and\s+([0-9.]+)\s*(\S+)", s, flags=re.IGNORECASE)
    if m:
        lo = _conc_as_nM(float(m.group(1)), m.group(2))
        hi = _conc_as_nM(float(m.group(3)), m.group(4))
        if lo is None or hi is None:
            return _unparsed("Dmax_conc", raw, _EMPTY.copy())
        return {"value": None, "operator": None, "range_min": lo, "range_max": hi}

    m = re.fullmatch(r"(?:~\s*)?([0-9.]+)\s*(?:-|to)\s*([0-9.]+)\s*(\S+)", s, flags=re.IGNORECASE)
    if m:
        lo = _conc_as_nM(float(m.group(1)), m.group(3))
        hi = _conc_as_nM(float(m.group(2)), m.group(3))
        if lo is None or hi is None:
            return _unparsed("Dmax_conc", raw, _EMPTY.copy())
        mid = (float(lo) + float(hi)) / 2
        mid = int(mid) if mid.is_integer() else mid
        return {"value": mid, "operator": None, "range_min": lo, "range_max": hi}

    m = re.fullmatch(
        r"(<=|>=|<|>|~|up\s*to|at\s*most|at\s*least|more\s*than|less\s*than|over|under|"
        r"below|above|around|about|approximately|roughly|near(?:ly)?|approaching)?\s*"
        r"([0-9.]+)\s*(\S+)",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        word = re.sub(r"\s+", "", (m.group(1) or "")).lower()
        operator = {
            "": None, "<": "<", "<=": "<=", ">": ">", ">=": ">=", "~": "~",
            "upto": "<=", "atmost": "<=", "lessthan": "<", "under": "<", "below": "<",
            "atleast": ">=", "morethan": ">", "over": ">", "above": ">",
        }.get(word, "~")
        val = _conc_as_nM(float(m.group(2)), m.group(3))
        if val is None:
            return _unparsed("Dmax_conc", raw, _EMPTY.copy())
        return {"value": val, "operator": operator, "range_min": None, "range_max": None}

    return _unparsed("Dmax_conc", raw, _EMPTY.copy())


def _normalize_conc(v):
    if v is None or pd.isna(v):
        return pd.NA
    val = float(v)
    return int(val) if val.is_integer() else val


def standardize_dmax_conc(df_llm: pd.DataFrame, input_csv: str) -> pd.DataFrame:
    dmax_conc_raw = pd.read_csv(input_csv, usecols=["Dmax_conc"], dtype=str)["Dmax_conc"]
    parsed = dmax_conc_raw.apply(parse_dmax_conc).apply(pd.Series)

    for col in ["value", "range_min", "range_max"]:
        parsed[col] = parsed[col].map(_normalize_conc)

    conc_values = pd.concat([parsed["value"], parsed["range_min"], parsed["range_max"]]).dropna()
    has_fraction = conc_values.apply(lambda v: not float(v).is_integer()).any()
    conc_dtype = object if has_fraction else "Int64"

    df_llm["Dmax_conc"] = parsed["value"].astype(conc_dtype)
    df_llm["Dmax_conc_operator"] = parsed["operator"]
    df_llm["Dmax_conc_range_min"] = parsed["range_min"].astype(conc_dtype)
    df_llm["Dmax_conc_range_max"] = parsed["range_max"].astype(conc_dtype)
    df_llm["Dmax_conc_parse_status"] = dmax_conc_raw.map(
        lambda raw: _activity_parse_status("Dmax_conc", raw)
    )

    has_conc = (
        df_llm["Dmax_conc"].notna()
        | df_llm["Dmax_conc_range_min"].notna()
        | df_llm["Dmax_conc_range_max"].notna()
    )
    df_llm["Dmax_conc_units"] = pd.NA
    df_llm.loc[has_conc, "Dmax_conc_units"] = "nM"

    df_llm = reorder_standardized_columns(df_llm)
    return df_llm


def standardize_smiles(df: pd.DataFrame) -> pd.DataFrame:
    df, _ = canonicalize_smiles_column(df, column="SMILES")
    df = add_compound_key(df)
    df["Connectivity_Key"] = df["Compound_Key"].str[:14]
    df = _place_after(df, "SMILES", ["Compound_Key", "Connectivity_Key"])
    return df


def add_autotpd_id(df: pd.DataFrame, prefix: str, width: int = 6) -> pd.DataFrame:
    df = df.drop(columns=["AutoTPD_ID"], errors="ignore")
    df.insert(0, "AutoTPD_ID", [f"{prefix}-{i:0{width}d}" for i in range(1, len(df) + 1)])
    print(f"AutoTPD_ID minted: {len(df)} rows, {df['AutoTPD_ID'].iloc[0]} .. {df['AutoTPD_ID'].iloc[-1]}")
    return df

ASSAY_STD_MAP = {

    "Western blot":                                                   "Western blot",
    "Western Blot":                                                   "Western blot",
    "western blot":                                                   "Western blot",
    "Western blotting":                                               "Western blot",
    "Western blotting (densitometry)":                                "Western blot",
    "quantitative western blotting":                                  "Western blot",
    "Licor-quantification of Western blots":                           "Western blot",
    "quantitative western blot (immunoblot)":                         "Western blot",
    "quantitative fluorescence western blot":                         "Western blot",
    "western blot quantification":                                    "Western blot",
    "Western blot quantification":                                    "Western blot",
    "western blot (time-course immunoblot)":                          "Western blot",
    "Western blot densitometry":                                      "Western blot",
    "Western blot (densitometry)":                                    "Western blot",
    "Western blot densitometry (immunoblot band intensity)":          "Western blot",
    "Western blot (grayscale quantification)":                        "Western blot",
    "Western blot (SDS-PAGE)":                                        "Western blot",
    "Western blot (S100A4 level)":                                    "Western blot",
    "Western blot (soluble/insoluble fractionation)":                 "Western blot",
    "Western blot (+elacridar pretreatment)":                         "Western blot",
    "Western blot (\u2212elacridar pretreatment)":                     "Western blot",
    "Western blot (immunoblot)":                                      "Western blot",
    "Western Blot(Immunoblot)":                                       "Western blot",
    "Western blot (immunoblot) densitometry":                         "Western blot",
    "Western blot (immunoblot) dose\u2013response":                    "Western blot",
    "Western blot (immunoblot) dose\u2013response quantification":     "Western blot",
    "Western blot (immunoblot), densitometric quantification":        "Western blot",
    "Western blot (immunoblot), densitometry (ImageJ)":               "Western blot",
    "immunoblot":                                                     "Western blot",
    "Immunoblot":                                                     "Western blot",
    "immunoblotting":                                                 "Western blot",
    "Immunoblotting":                                                 "Western blot",
    "immunoblot (western blot)":                                      "Western blot",
    "Immunoblot (Western blot)":                                      "Western blot",
    "immunoblotting (Western blot)":                                  "Western blot",
    "immunoblotting (western blot)":                                  "Western blot",
    "Immunoblotting (Western blot)":                                  "Western blot",
    "Immunoblotting (western blot)":                                  "Western blot",
    "Immunoblot densitometry (Western blot)":                         "Western blot",
    "Immunoblot densitometry of cyclin K levels":                     "Western blot",
    "Immunoblot (western blot) densitometry normalized to \u03b2-tubulin": "Western blot",
    "immunoblot (western blot) quantification of IDO1 protein levels": "Western blot",
    "In vivo tumor immunoblot (Western blot)":                        "Western blot",

    "capillary electrophoresis":                                      "Capillary immunoassay",
    "capillary electrophoresis (WES)":                                "Capillary immunoassay",
    "Capillary-based immunoassay (Simple Western)":                   "Capillary immunoassay",
    "Simple Western immunoassay (capillary-based)":                   "Capillary immunoassay",
    "Automated capillary Western blot (JESS)":                        "Capillary immunoassay",

    "HiBiT":                                                          "HiBiT",
    "HiBit":                                                          "HiBiT",
    "HiBiT assay":                                                    "HiBiT",
    "HiBiT luminescence":                                             "HiBiT",
    "HiBiT luminescence assay":                                       "HiBiT",
    "HiBiT luminescence (Firefly-normalized)":                        "HiBiT",
    "HiBiT degradation assay":                                        "HiBiT",
    "HiBiT degradation assay (+ MLN4924)":                            "HiBiT",
    "HiBiT(with MLN4924)":                                            "HiBiT",
    "HiBiT lytic luminescence assay":                                 "HiBiT",
    "HiBiT live-cell luminescent degradation assay":                  "HiBiT",
    "HiBiT-LgBiT live-cell luminescence assay (%ARD)":                "HiBiT",
    "HiBiT luciferase CK1\u03b1 degradation assay":                    "HiBiT",
    "HiBiT luciferase degradation assay (Nano-Glo HiBiT Lytic Detection)": "HiBiT",
    "NanoGlo HiBiT lytic assay":                                      "HiBiT",
    "IKZF1-HiBiT kinetic degradation assay":                          "HiBiT",
    "GSPT1-HiBiT endpoint degradation assay":                         "HiBiT",

    "NanoLuc degradation kinetics":                                   "NanoLuc reporter",
    "NanoLuc reporter assay (BTK-nLuc), endpoint":                    "NanoLuc reporter",
    "NanoLuc reporter assay (IKZF1-nLuc), endpoint":                  "NanoLuc reporter",
    "NanoLuc reporter assay (IKZF3-nLuc), endpoint":                  "NanoLuc reporter",

    "MSD":                                                            "MSD",
    "MSD assay":                                                      "MSD",
    "MSD degrader assay":                                             "MSD",
    "ELISA-based MSD assay":                                          "MSD",
    "MSD (+elacridar pretreatment)":                                  "MSD",
    "MSD (\u2212elacridar pretreatment)":                              "MSD",

    "ELISA":                                                          "ELISA",
    "in-cell ELISA":                                                  "ELISA",

    "AlphaLISA":                                                      "AlphaLISA",

    "targeted mass spectrometry":                                     "Mass spectrometry",

    "HTRF":                                                           "HTRF",

    "Flow cytometry":                                                 "Flow cytometry",
    "flow cytometry":                                                 "Flow cytometry",
    "FACS":                                                           "Flow cytometry",
    "Flow cytometry(FACS)":                                           "Flow cytometry",
    "Flow cytometry(eGFP/mCherry)":                                   "Flow cytometry",
    "Flow cytometry (eGFP:mCherry reporter)":                         "Flow cytometry",
    "EGFP/mCherry reporter flow cytometry":                           "Flow cytometry",
    "FACS dual fluorescence reporter assay":                          "Flow cytometry",
    "Flow cytometry reporter assay (BRD4BD2-eGFP) measuring eGFP/mCherry ratio": "Flow cytometry",
    "Flow cytometry of NS3-eGFP-2A-mCherry reporter (eGFP/mCherry ratio)":       "Flow cytometry",
    "flow cytometry (BTK-GFP/mCherry ratiometric)":                   "Flow cytometry",
    "flow cytometry (anti-AXL-PE)":                                   "Flow cytometry",
    "flow cytometry (anti-MERTK-PE)":                                 "Flow cytometry",
    "flow cytometry (anti-EGFR-PE on EGFR/TAM chimera)":              "Flow cytometry",

    "Immunofluorescence":                                             "Fluorescence imaging",
    "immunofluorescence assay":                                       "Fluorescence imaging",
    "immunofluorescence imaging":                                     "Fluorescence imaging",
    "Live-cell fluorescence imaging of GFP-BRD4":                     "Fluorescence imaging",
    "Imaging-based GFP/mCherry ratio (high-content imaging, Operetta)": "Fluorescence imaging",
    "FluoReSyn nanobody biosensor (GFP immunostaining)":              "Fluorescence imaging",

    "eGFP/mCherry":                                                   "Fluorescent reporter",
    "EGFP/mCherry reporter assay":                                    "Fluorescent reporter",
    "BRD4BD2-eGFP/mCherry reporter assay":                            "Fluorescent reporter",
    "HEK293T cyclin K\u2013eGFP reporter assay":                       "Fluorescent reporter",

    "Firefly luciferase activity assay":                              "Luciferase reporter",
    "CCNK-luc reporter (luciferase) degradation assay":               "Luciferase reporter",

    "ProLabel":                                                       "Enzyme fragment complementation",
    "In Cell Western":                                                "In-cell immunoassay",
    "Cytoblot":                                                       "In-cell immunoassay",
    "Dot blot":                                                       "Dot blot",
    "protein degradation profiling":                                  "protein degradation profiling",
    "ELISA + Western Blot":                                           "ELISA + Western Blot",
    "BRD4BD2-eGFP reporter degradation (flow cytometry)":             "Flow cytometry",
    "BRD9-HiBiT endpoint degradation assay (NanoGlo Lytic)":          "HiBiT",
    "cell-based degradation assay":                                   "Cell-based degradation assay",
    "CK1α HiBiT assay":                                               "HiBiT",
    "cycKeGFP reporter assay":                                        "Fluorescent reporter",
    "DEFUSE system rescue assay (AP1903-induced death rescue)":       "Functional rescue assay",
    "EGFP-IRES-mCherry degradation reporter (flow cytometry)":        "Flow cytometry",
    "FACS reporter assay (BRD4 dual fluorescence protein stability reporter)": "Flow cytometry",
    "Flow cytometry (dual-reporter ZF-EGFP/mCherry ratio)":           "Flow cytometry",
    "flow cytometry (GFP reporter)":                                  "Flow cytometry",
    "Flow cytometry BRD4BD2-eGFP reporter degradation":               "Flow cytometry",
    "HiBiT cellular degradation assay":                               "HiBiT",
    "HiBiT endpoint degradation assay":                               "HiBiT",
    "HiBiT LCK degradation assay (Nano-Glo HiBiT)":                   "HiBiT",
    "HiBiT LCK degradation assay (Nano-Glo HiBiT) with 10 μM lenalidomide cotreatment": "HiBiT",
    "HiBiT live-cell kinetic degradation assay":                      "HiBiT",
    "HiBiT luciferase degradation assay":                             "HiBiT",
    "HiBiT luminescence-based degradation assay":                     "HiBiT",
    "HiBiT luminescent degradation assay":                            "HiBiT",
    "HiBiT luminescent protein quantification assay":                 "HiBiT",
    "HiBiT luminescent protein quantification assay (Nano-Glo HiBiT lytic)": "HiBiT",
    "HiBiT lytic assay (Promega)":                                    "HiBiT",
    "HiBiT lytic degradation assay":                                  "HiBiT",
    "HiBiT protein degradation assay":                                "HiBiT",
    "HiBiT quantification (ZBTB11-HiBiT reporter)":                    "HiBiT",
    "HiBiT split-luciferase degradation assay":                       "HiBiT",
    "HiBiT-based cell degradation assay":                             "HiBiT",
    "HiBiT-BRD4 degradation assay (endogenous)":                      "HiBiT",
    "HiBiT-LgBiT live-cell luminescence time-course (Nano-Glo Endurazine)": "HiBiT",
    "HiBiT-RBM39-HA lytic bioluminescence assay":                     "HiBiT",
    "HiBiT-RR endogenous VAV1 degradation assay":                     "HiBiT",
    "HiBiT–BRD4 cellular degradation assay (Nano-Glo HiBiT lytic)":    "HiBiT",
    "HiBiT–ENL degradation assay (luminescence)":                      "HiBiT",
    "HTRF (TR-FRET sandwich immunoassay)":                            "HTRF",
    "IKZF1 HiBiT assay":                                              "HiBiT",
    "IKZF1-HiBiT luminescence degradation assay":                     "HiBiT",
    "IKZF3-HiBiT luminescence degradation assay":                     "HiBiT",
    "Immunoblot (dose–response quantification of endogenous HuR)":     "Western blot",
    "immunoblot (Western blot)":                                      "Western blot",
    "Immunoblotting (western blot) quantification after IFN-γ washout": "Western blot",
    "Immunofluorescence (anti-BRD9)":                                 "Fluorescence imaging",
    "live-cell flow cytometry":                                       "Flow cytometry",
    "NanoGlo luciferase assay (f-Myc-Nluc fusion)":                    "NanoLuc reporter",
    "NanoGlo luminescence assay":                                     "NanoLuc reporter",
    "Ratiometric flow cytometry (EGFP/mCherry)":                      "Flow cytometry",
    "Western blot (immunoblot) quantification":                       "Western blot",
    "Western blot degradation assay":                                 "Western blot",
    "western blot densitometry":                                      "Western blot",
    "Western blot densitometry of cyclin K levels":                   "Western blot",
    "Western blot densitometry of intracellular capsid abundance":    "Western blot",
    "Western blot densitometry of target abundance":                  "Western blot",
    "Western blot of BRD4 levels; DC50 from dose-response fit (Hill slope = 1)": "Western blot",
    "western blotting (dose–response)":                                  "Western blot",
    "AR HiBiT assay": "HiBiT",
    "Activity Assays": None,
    "AlphaLISA SureFire Ultra (cell lysate)": "AlphaLISA",
    "AlphaLISA SureFire Ultra (cell lysate; co-treated with 3 μM MG132)": "AlphaLISA",
    "AlphaLISA cellular degradation assay": "AlphaLISA",
    "Aurora A HiBiT assay": "HiBiT",
    "BRD4 domain fluorescent reporter flow cytometry assay": "Flow cytometry",
    "CEIA": "Capillary immunoassay",
    "Cell-free proteasomal degradation assay with yeast 26S proteasome; Western blot quantification": "Western blot",
    "CellTiter-Glo": None,
    "Cellular degradation": "Cell-based degradation assay",
    "Cellular fluorescence assay (co-expression of USP39-OFP and VHL_WT-EGFP)": "Fluorescence imaging",
    "DIA-MS proteomics": "Mass spectrometry",
    "EGFP-MDM2 fluorescence imaging": "Fluorescence imaging",
    "EGFP/mCherry reporter assay (flow cytometry)": "Flow cytometry",
    "EGFP–LIMK2 live-cell kinetic degradation assay (Incucyte)": "Fluorescence imaging",
    "ELISA; Western blot": "ELISA + Western Blot",
    "ELISA; western blot": "ELISA + Western Blot",
    "ERα high-throughput imaging degradation assay": "Fluorescence imaging",
    "ERα nanoluciferase degradation assay": "NanoLuc reporter",
    "Endogenous WDR5 degradation assay (ProteinSimple Jess)": "Capillary immunoassay",
    "Endpoint HiBiT lytic luminescence assay": "HiBiT",
    "FKBP12-HiBiT degradation assay (dose-response, luminescence)": "HiBiT",
    "Firefly luciferase luminescence": "Luciferase reporter",
    "Flow cytometry (EGFP levels)": "Flow cytometry",
    "Flow cytometry (EGFP median intensity)": "Flow cytometry",
    "Flow cytometry (EGFP signal)": "Flow cytometry",
    "Flow cytometry (EGFR/TAM chimeric reporter, anti-EGFR-PE)": "Flow cytometry",
    "Flow cytometry (GFP fluorescence), LNP K1 delivery": "Flow cytometry",
    "Flow cytometry (GFP fluorescence), Lipofectamine 2000 delivery": "Flow cytometry",
    "Flow cytometry (GFP intensity)": "Flow cytometry",
    "Flow cytometry (Halotag Janelia 647 probe) for DC50; Western blot for Dmax": "Flow cytometry",
    "Flow cytometry (surface AXL, anti-AXL-PE)": "Flow cytometry",
    "Flow cytometry (surface MERTK, anti-MERTK-PE)": "Flow cytometry",
    "Flow cytometry in human PBMC lymphocyte and monocyte subsets": "Flow cytometry",
    "Flow cytometry of BTK-GFP/mCherry reporter": "Flow cytometry",
    "Flow cytometry of surface CAR expression (anti-mouse IgG staining; MFI-based quantification)": "Flow cytometry",
    "HiBiT BRD4 degradation assay": "HiBiT",
    "HiBiT DLR assay": "HiBiT",
    "HiBiT System": "HiBiT",
    "HiBiT cellular degradation assay (Nano-Glo HiBiT lytic)": "HiBiT",
    "HiBiT degradation assay (HiBiT/CTG)": "HiBiT",
    "HiBiT end-point BRD4 degradation assay (luminescence)": "HiBiT",
    "HiBiT live-cell luminescence degradation assay": "HiBiT",
    "HiBiT luciferase assay": "HiBiT",
    "HiBiT luciferase degradation assay (Nano-Glo HiBiT Lytic Detection System)": "HiBiT",
    "HiBiT luminescence (co-treatment with GSK2606414)": "HiBiT",
    "HiBiT luminescence (co-treatment with PDD17273)": "HiBiT",
    "HiBiT luminescence assay (BRD4-HiBiT)": "HiBiT",
    "HiBiT luminescence assay (Promega Nano-Glo HiBiT Lytic)": "HiBiT",
    "HiBiT luminescence degradation assay": "HiBiT",
    "HiBiT luminescence degradation assay (Nano-Glo HiBiT Lytic Detection System)": "HiBiT",
    "HiBiT lytic assay": "HiBiT",
    "HiBiT lytic cellular assay": "HiBiT",
    "HiBiT lytic detection (endpoint luminescence)": "HiBiT",
    "HiBiT lytic detection luminescence assay": "HiBiT",
    "HiBiT split-luciferase assay": "HiBiT",
    "HiBiT system": "HiBiT",
    "HiBiT-BRD4 Degradation Assay": "HiBiT",
    "HiBiT-BRD4 Nanoluciferase degradation assay": "HiBiT",
    "HiBiT-BRD4 assay": "HiBiT",
    "HiBiT-BRD4 degradation assay": "HiBiT",
    "HiBiT-BRD4 luciferase assay": "HiBiT",
    "HiBiT-FBXO22 luciferase assay": "HiBiT",
    "HiBiT-HIPK4 (split-luciferase) degradation assay": "HiBiT",
    "HiBiT-TEAD1 degradation assay (HiBiT luminescence normalized to CTG)": "HiBiT",
    "HiBiT-based cellular degradation assay (NanoLuc luminescence)": "HiBiT",
    "HiBiT-tagged WDR5 degradation assay (exogenous; NanoLuc/HiBiT)": "HiBiT",
    "HiBit BRD4 degradation": "HiBiT",
    "HiBit assay": "HiBiT",
    "High-content confocal immunofluorescence of nuclear BRD4 puncta": "Fluorescence imaging",
    "High-content imaging (Operetta) GFP/mCherry ratio assay": "Fluorescence imaging",
    "Immunoblot (IB) quantification": "Western blot",
    "Immunoblot (Western blot) densitometry of hRpn13Pru normalized to β-actin": "Western blot",
    "Immunoblot (Western blot) of CDK9; band intensity quantified by ImageJ": "Western blot",
    "Immunoblot (Western blot) quantification of target abundance": "Western blot",
    "Immunoblot (tumor KRAS(G12D))": "Western blot",
    "Immunoblot (western blot)": "Western blot",
    "Immunoblot (western blot) densitometry": "Western blot",
    "Immunoblot (western blot) with densitometry": "Western blot",
    "Immunoblot analysis (western blot)": "Western blot",
    "Immunoblot densitometry": "Western blot",
    "Immunoblotting (IB)": "Western blot",
    "Immunoblotting (Western blot) densitometry": "Western blot",
    "Immunoblotting-based endogenous degradation assay": "Western blot",
    "Immunofluorescence (anti-HiBiT)": "Fluorescence imaging",
    "Immunofluorescence microscopy quantification (HeLa-USP39_OFP; no MLN4924)": "Fluorescence imaging",
    "Immunofluorescence microscopy quantification (HeLa-USP39_OFP; with MLN4924 5 µM)": "Fluorescence imaging",
    "Immunofluorescence quantitation of BRD4": "Fluorescence imaging",
    "In vitro DC50 (degradation)": "Cell-based degradation assay",
    "In-Cell ELISA (KRAS(G12D) degradation)": "ELISA",
    "In-Cell Western (ICW)": "In-cell immunoassay",
    "JESS automated immunoblotting (Jess assay)": "Capillary immunoassay",
    "LI-COR Western blot quantification": "Western blot",
    "Live-cell HiBiT kinetic degradation assay": "HiBiT",
    "Live-cell kinetic HiBiT luminescence degradation assay": "HiBiT",
    "Luminescence assay (NanoLuc/Vivazine)": "NanoLuc reporter",
    "MSD (electrochemiluminescence immunoassay)": "MSD",
    "MSD sandwich ELISA (10-point dose-response), 4 h": "MSD",
    "NS3-eGFP-2A-mCherry reporter flow cytometry (eGFP/mCherry ratio)": "Flow cytometry",
    "NSD2-HiBiT assay (Nano-Glo HiBiT Lytic)": "HiBiT",
    "Nano-Glo HiBiT lytic detection assay": "HiBiT",
    "NanoBRET degradation assay (NanoLuc luminescence)": "NanoLuc reporter",
    "NanoBiT": "HiBiT",
    "NanoLuc live cell assay (time-course)": "NanoLuc reporter",
    "NanoLuc live-cell reporter bioluminescence (nLucK0-RIPK1KD fusion)": "NanoLuc reporter",
    "NanoLuc live-cell reporter bioluminescence (nLucWT-RIPK1KD fusion)": "NanoLuc reporter",
    "NanoLuc luciferase assay": "NanoLuc reporter",
    "NanoLuc luciferase assay (C-terminal fusion)": "NanoLuc reporter",
    "NanoLuc luciferase assay (N-terminal fusion)": "NanoLuc reporter",
    "NanoLuc luciferase assay (TNKS1-NanoLuc reporter)": "NanoLuc reporter",
    "NanoLuc luciferase assay (nLuc-RIPK1)": "NanoLuc reporter",
    "NanoLuc luciferase assay + MG-132": "NanoLuc reporter",
    "NanoLuc luciferase assay + MLN4924": "NanoLuc reporter",
    "NanoLuc luminescence assay": "NanoLuc reporter",
    "NanoLuc reporter (endpoint)": "NanoLuc reporter",
    "PTPN2 HiBiT degradation assay": "HiBiT",
    "Promega Nano-Glo HiBiT assay": "HiBiT",
    "Real-time live-cell HiBiT luminescence": "HiBiT",
    "SHP2-sfGFPxK loss-of-signal by flow cytometry (GFP MFI)": "Flow cytometry",
    "STAT3 HiBiT assay": "HiBiT",
    "STAT3-HiBiT cellular assay": "HiBiT",
    "Simple Western": "Capillary immunoassay",
    "Simple Western (JESS) immunoassay": "Capillary immunoassay",
    "Targeted proteomic analysis (PRM-LC-MS)": "Mass spectrometry",
    "Triton X-100 fractionation assay (Western blot)": "Western blot",
    "Western Blot KAT6A": "Western blot",
    "Western Blot and ELISA": "ELISA + Western Blot",
    "Western Blot and targeted proteomics": "Western blot",
    "Western blot (CHX chase)": "Western blot",
    "Western blot (HaloTag7 immunoblot)": "Western blot",
    "Western blot (IRAK4/GAPDH quantification)": "Western blot",
    "Western blot (ImageJ quantification)": "Western blot",
    "Western blot (ProteinSimple WES)": "Capillary immunoassay",
    "Western blot (SDS-PAGE followed by immunoblot)": "Western blot",
    "Western blot (SDS-PAGE immunoblot quantitation of RIPK2)": "Western blot",
    "Western blot (WB)": "Western blot",
    "Western blot (WB) densitometry": "Western blot",
    "Western blot (anti-BTK) with densitometric quantitation": "Western blot",
    "Western blot (detergent solubility fractionation)": "Western blot",
    "Western blot (dose-response quantification)": "Western blot",
    "Western blot (dose-response)": "Western blot",
    "Western blot (dose–response)": "Western blot",
    "Western blot (immunoblot band intensity)": "Western blot",
    "Western blot (immunoblot) densitometry (ImageJ)": "Western blot",
    "Western blot (immunoblot) densitometry of hRpn13Pru normalized to β-actin": "Western blot",
    "Western blot (immunoblot) densitometry; MA203 dose–response with 0.5 mM HU for 24 h; nonlinear regression": "Western blot",
    "Western blot (immunoblot) densitometry; MA203 dose–response with 0.5 mM HU for 8 h (within a 24 h MA203 treatment); nonlinear regression": "Western blot",
    "Western blot (immunoblot) densitometry; MA203 dose–response with 2 µM Ara-C for 8 h (within a 24 h MA203 treatment); nonlinear regression": "Western blot",
    "Western blot (immunoblot) densitometry; MA203 dose–response without co-treatment; nonlinear regression": "Western blot",
    "Western blot (immunoblot) dose-response quantification": "Western blot",
    "Western blot (immunoblot) dose–response; band intensity quantified": "Western blot",
    "Western blot (immunoblot) dose–response; band intensity quantified (ImageJ)": "Western blot",
    "Western blot (immunoblot) of AR protein levels": "Western blot",
    "Western blot (immunoblot) quantification of KRAS levels": "Western blot",
    "Western blot (immunoblot) quantification of NUDT1 levels": "Western blot",
    "Western blot (immunoblot) quantification of Pin1": "Western blot",
    "Western blot (immunoblot) quantification of SRC-1 levels": "Western blot",
    "Western blot (immunoblotting)": "Western blot",
    "Western blot (immunoblotting) for E protein depletion": "Western blot",
    "Western blot (immunoblotting) quantification of BRD4": "Western blot",
    "Western blot (kinetics)": "Western blot",
    "Western blot (protein quantification)": "Western blot",
    "Western blot (quantified band intensities)": "Western blot",
    "Western blot (quantified band intensity)": "Western blot",
    "Western blot (quantified)": "Western blot",
    "Western blot analysis": "Western blot",
    "Western blot analysis (time-course)": "Western blot",
    "Western blot analysis of total HDAC8 levels": "Western blot",
    "Western blot analysis; hypoxia (<0.1% O2)": "Western blot",
    "Western blot analysis; normoxia (21% O2)": "Western blot",
    "Western blot degradation assay (dose–response)": "Western blot",
    "Western blot densitometric analysis of CDK6 band intensities": "Western blot",
    "Western blot densitometry (IFNγ-induced cells; 24 h treatment)": "Western blot",
    "Western blot densitometry (ImageJ)": "Western blot",
    "Western blot densitometry (cytoplasmic fraction)": "Western blot",
    "Western blot densitometry of 3CL-flag": "Western blot",
    "Western blot densitometry of hRpn13Pru normalized to β-actin": "Western blot",
    "Western blot densitometry, LNP K1 delivery": "Western blot",
    "Western blot densitometry; DFX-induced hypoxia (10 μM)": "Western blot",
    "Western blot densitometry; hypoxia at 3% O2": "Western blot",
    "Western blot of BTK levels; normalization to β-actin; DC50/Dmax by second-order polynomial fit (Prism)": "Western blot",
    "Western blot quantification of NUDT5 protein levels": "Western blot",
    "Western blot quantification of PRMT7 levels after compound treatment": "Western blot",
    "Western blot quantification of RIPK1 levels": "Western blot",
    "Western blot with grayscale quantification": "Western blot",
    "Western blot-based cellular degradation assay": "Western blot",
    "Western blotting (PAK4 protein levels)": "Western blot",
    "Western blotting (WB)": "Western blot",
    "Western blotting (dose–response)": "Western blot",
    "Western blotting analysis": "Western blot",
    "Western blotting dose-response of DAPK1 levels": "Western blot",
    "capillary western (Simple Western JESS)": "Capillary immunoassay",
    "capillary-based immunoassay": "Capillary immunoassay",
    "capillary-based immunoassay (Simple Western)": "Capillary immunoassay",
    "cell-based BRM/BRG1 HiBiT assay": "HiBiT",
    "cell-based BTK degradation": "Cell-based degradation assay",
    "cell-based BTK degradation assay": "Cell-based degradation assay",
    "cell-based DC50 determination; assay not specified": "Cell-based degradation assay",
    "cell-based ER degradation (in vitro)": "Cell-based degradation assay",
    "cell-based ER degradation (in vitro; with CDK4/6 inhibitor)": "Cell-based degradation assay",
    "cell-based degradation assay (ATTEC)": "Cell-based degradation assay",
    "cell-based degradation assay (method not specified)": "Cell-based degradation assay",
    "cell-based degradation assay in HEK293T cells transduced with α-syn preformed fibrils (method not specified)": "Cell-based degradation assay",
    "cell-based degradation assay in α-synuclein-overexpressing HEK293T cells (method not specified)": "Cell-based degradation assay",
    "cell-based degradation in prostate cancer cell lines (unspecified method)": "Cell-based degradation assay",
    "cell-based viral replication degradation assay": "Cell-based degradation assay",
    "cell-free ClpC1-NTD degradation assay (capillary Western, WES)": "Capillary immunoassay",
    "cell-free ClpC1-NTD degradation assay (capillary Western, WES; M. smegmatis machinery)": "Capillary immunoassay",
    "cell-free ClpC1-NTD degradation assay (capillary Western, WES; M. tuberculosis machinery)": "Capillary immunoassay",
    "cell-free amplified luminescence proximity homogeneous assay": "AlphaLISA",
    "cellular BRD4 degradation assay": "Cell-based degradation assay",
    "cellular BRD4 degradation assay with X-ray irradiation": "Cell-based degradation assay",
    "cellular BTK degradation assay (method not specified)": "Cell-based degradation assay",
    "cellular ClpC1 degradation (capillary Western, JESS TPN)": "Capillary immunoassay",
    "cellular PD-L1 degradation assay (unspecified method)": "Cell-based degradation assay",
    "cellular degradation assay": "Cell-based degradation assay",
    "cellular degradation assay (literature)": "Cell-based degradation assay",
    "cellular degradation assay (method not specified)": "Cell-based degradation assay",
    "cellular degradation assay (proteasome-dependent)": "Cell-based degradation assay",
    "degradation assay": "Cell-based degradation assay",
    "dose-response degradation assay": "Cell-based degradation assay",
    "dose–response degradation assay": "Cell-based degradation assay",
    "endogenous protein immunoblot (western blot) dose–response": "Western blot",
    "enzyme fragment complementation (EFC) targeted protein degradation assay": "Enzyme fragment complementation",
    "flow cytometry (EGFP fluorescence MFI normalized to mCherry)": "Flow cytometry",
    "flow cytometry (EGFP fluorescence MFI)": "Flow cytometry",
    "flow cytometry stability reporter (dose/time course)": "Flow cytometry",
    "flow cytometry stability reporter (time course)": "Flow cytometry",
    "fluorescence-based live cell imaging (GFP)": "Fluorescence imaging",
    "high-throughput immunofluorescence assay": "Fluorescence imaging",
    "immunoblot (Western blot) dose–response": "Western blot",
    "immunoblot (after 660 nm irradiation)": "Western blot",
    "immunoblot (dark)": "Western blot",
    "immunoblot (dose–response)": "Western blot",
    "immunoblot (western blot) densitometry": "Western blot",
    "immunoblot densitometry (western blot)": "Western blot",
    "immunoblotting (HiBiT)": "Western blot",
    "immunoblotting (Western blot) quantification": "Western blot",
    "immunoblotting (Western blot) quantification of AR levels": "Western blot",
    "immunoblotting (Western blot) with densitometric quantification": "Western blot",
    "immunoblotting (western blot) quantification": "Western blot",
    "immunoblotting (western blot) with densitometry": "Western blot",
    "immunoblotting studies": "Western blot",
    "immunofluorescence": "Fluorescence imaging",
    "immunofluorescence imaging (anti-HA), automated image analysis": "Fluorescence imaging",
    "immunostaining and microscopy": "Fluorescence imaging",
    "in vitro assays with human monocytes": None,
    "in vitro cell-based degradation assay": "Cell-based degradation assay",
    "in vitro cellular degradation (method not specified)": "Cell-based degradation assay",
    "in-cell western immunofluorescence": "In-cell immunoassay",
    "live-cell flow cytometry (PIP-Fc staining of surface integrins)": "Flow cytometry",
    "live-cell flow cytometry (cell-surface EGFR)": "Flow cytometry",
    "mesoscale discovery assay": "MSD",
    "multiplexed western blotting": "Western blot",
    "not specified": None,
    "not specified (company product sheet)": None,
    "quantitative Western blot": "Western blot",
    "quantitative fluorescent western blotting": "Western blot",
    "quantitative immunoblot (Odyssey)": "Western blot",
    "quantitative immunofluorescence": "Fluorescence imaging",
    "quantitative western blot (immunoblotting)": "Western blot",
    "real-time HiBiT detection assay": "HiBiT",
    "targeted MS": "Mass spectrometry",
    "unspecified": None,
    "western blot (immunoblot) quantification": "Western blot",
    "western blot (relative quantification)": "Western blot",
    "western blot (total EGFR)": "Western blot",
    "western blot (total HER2)": "Western blot",
    "western blotting": "Western blot",
    "western blotting (densitometry)": "Western blot",
    "Flow cytometry; Western blot":                                   "Flow cytometry + Western blot",
    "flow cytometry; Western blotting (WB)":                          "Flow cytometry + Western blot",
    "in vitro mSA degradation (SDS-PAGE/Coomassie densitometry)":     "Gel densitometry",
    "in-gel fluorescence":                                            "Gel densitometry",
    "Autophagic flux assay":                                          None,
    "cell-free assay":                                                None,
    "in vivo tumor ER degradation (MCF7 CDX)":                        None,
    "in vivo tumor ER degradation (PDX)":                             None,
}


def standardize_assay(df: pd.DataFrame) -> pd.DataFrame:
    if "Assay" not in df.columns:
        print("Column 'Assay' not present, skipping assay standardization")
        return df

    stripped = df["Assay"].astype("string").str.strip()
    df["Assay_STD"] = stripped.map(
        lambda a: ASSAY_STD_MAP.get(a) if pd.notna(a) and a != "" else None
    ).astype("string")

    n_rows = (stripped.notna() & (stripped != "")).sum()
    n_std = df["Assay_STD"].notna().sum()
    print(f"Rows with an assay: {n_rows}/{len(df)}")
    print(f"Assay_STD mapped: {n_std}/{n_rows} rows")

    seen = sorted(set(stripped[(stripped.notna()) & (stripped != "") & df["Assay_STD"].isna()]))
    unmapped = [a for a in seen if a not in ASSAY_STD_MAP]
    if len(seen) - len(unmapped):
        print(f"Assay values with no technique family by design: {len(seen) - len(unmapped)}")
    if unmapped:
        print(f"Assay values not in ASSAY_STD_MAP: {len(unmapped)}")
        for a in unmapped:
            print(f"  WARNING: '{a}' not in ASSAY_STD_MAP, Assay_STD left empty")
            _note_unmapped("assay", a)

    families = df["Assay_STD"].dropna().value_counts()
    if len(families):
        print(f"Technique families ({len(families)}):")
        for fam, n in families.items():
            print(f"  {n:>5}  {fam}")

    df = _place_after(df, "Assay", ["Assay_STD"])
    return df

_UNMAPPED_SOURCES = {

    "cell_line":              ("Cell_Line", "CL_SEARCH_MAP"),
    "cell_line_modification": ("Cell_Line", "CL_MODIFICATION_MAP"),
    "degradation_target":     ("Degradation_Target", "POI_LOOKUP"),
    "recruiter":              ("Recruiter", "E3_LOOKUP"),
    "assay":                  ("Assay", "ASSAY_STD_MAP"),
}
_EXAMPLE_COLUMNS = ["DOI", "Compound_Name", "Degradation_Target", "Recruiter", "Cell_Line", "Assay"]


def write_unmapped_report(df: pd.DataFrame, path: str, input_csv: str) -> None:
    import difflib
    import json

    items = []
    for kind in sorted(UNMAPPED):
        column, mapping_name = _UNMAPPED_SOURCES[kind]
        mapping = globals()[mapping_name]
        values = df[column].astype("string").str.strip()
        lowered = {str(key).lower(): key for key in mapping}
        for value in sorted(UNMAPPED[kind]):
            mask = (values == value).fillna(False)
            examples = (
                df.loc[mask, [c for c in _EXAMPLE_COLUMNS if c in df.columns]]
                .head(3).astype("string").fillna("").to_dict("records")
            )
            close = difflib.get_close_matches(value, [str(k) for k in mapping], n=3, cutoff=0.6)
            if value.lower() in lowered and lowered[value.lower()] not in close:
                close.insert(0, lowered[value.lower()])
            items.append({
                "kind": kind,
                "value": value,
                "mapping": mapping_name,
                "rows": int(mask.sum()),
                "examples": examples,
                "close_existing": [{"key": k, "value": mapping[k]} for k in close],
            })

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"input_csv": str(input_csv), "items": items}, indent=2, ensure_ascii=False,
    ) + "\n", encoding="utf-8")
    print(f"Unmapped values: {len(items)} written to {out_path}")


def write_unparsed_report(path: str, input_csv: str) -> None:
    import json

    source = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    items = []
    for field in sorted(UNPARSED):
        values = (
            source[field].astype("string").str.strip()
            if field in source.columns
            else pd.Series(dtype="string")
        )
        columns = [c for c in ["DOI", "Compound_Name", "Degradation_Target", field]
                   if c in source.columns]
        for value, status in sorted(UNPARSED[field].items()):
            mask = (values == value).fillna(False)
            items.append({
                "field": field,
                "value": value,
                "status": status,
                "rows": int(mask.sum()),
                "examples": source.loc[mask, columns].head(3).to_dict("records"),
            })

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"input_csv": str(input_csv), "items": items}, indent=2, ensure_ascii=False,
    ) + "\n", encoding="utf-8")
    unparsed = sum(1 for i in items if i["status"] == "unparsed")
    print(f"Unparsed activity values: {unparsed} needing review, "
          f"{len(items) - unparsed} recognised as non-numeric, written to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Standardize the extracted dataset."
    )
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT_CSV,
                        help="Path to the extracted CSV.")
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV,
                        help="Path to write the standardized CSV.")
    parser.add_argument("--record_id_prefix", type=str, default=None,
                        help="If given, mint a per-row AutoTPD_ID with this prefix, "
                             "e.g. --record_id_prefix AUTOTPD-PROTAC -> AUTOTPD-PROTAC-000001.")
    parser.add_argument("--unmapped_report", type=str, default=None,
                        help="If given, write values without a mapping entry to this JSON file.")
    parser.add_argument("--unparsed_report", type=str, default=None,
                        help="If given, write activity values no parser could read to this JSON file.")
    args = parser.parse_args()

    input_csv = args.input_csv
    output_csv = args.output_csv

    if not Path(input_csv).exists():
        print(f"Error: input CSV does not exist - {input_csv}")
        raise SystemExit(1)

    print(f"{'='*80}")
    print(f"Data standardization pipeline")
    print(f"  Input:  {input_csv}")
    print(f"  Output: {output_csv}")
    print(f"{'='*80}\n")

    df_llm = pd.read_csv(input_csv)

    print("\n--- Stage 1/6: Cell line standardization ---")
    df_llm, cl_info = standardize_cell_lines(df_llm)
    taxid_to_species = build_taxid_to_species(cl_info)

    print("\n--- Stage 2/6: Degradation target enrichment ---")
    df_llm = clean_and_enrich_poi(df_llm, taxid_to_species)

    print("\n--- Stage 3/6: Recruiter enrichment ---")
    df_llm = clean_and_enrich_e3(df_llm, taxid_to_species)

    print("\n--- Stage 4/6: DC50 / Dmax / Dmax_conc standardization ---")
    df_llm = standardize_dc50(df_llm, input_csv)
    df_llm = standardize_dmax(df_llm, input_csv)
    df_llm = standardize_dmax_conc(df_llm, input_csv)

    print("\n--- Stage 5/6: Assay standardization ---")
    df_llm = standardize_assay(df_llm)

    print("\n--- Stage 6/6: SMILES canonicalization ---")
    df_llm = standardize_smiles(df_llm)

    df_llm = _place_after(df_llm, "IUPAC_Name", ["IUPAC_Name_Source"])

    if args.record_id_prefix:
        print()
        df_llm = add_autotpd_id(df_llm, args.record_id_prefix)

    if args.unmapped_report:
        write_unmapped_report(df_llm, args.unmapped_report, input_csv)

    if args.unparsed_report:
        write_unparsed_report(args.unparsed_report, input_csv)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_llm.to_csv(out_path, index=False)

    print(f"\n{'='*80}")
    print(f"Standardization complete: {len(df_llm)} rows, {len(df_llm.columns)} columns")
    print(f"Saved to: {out_path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
