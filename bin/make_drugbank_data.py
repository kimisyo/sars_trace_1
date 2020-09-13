import xml.etree.ElementTree as ET

from rdkit import Chem
from rdkit.Chem import AllChem
from molvs.normalize import Normalizer, Normalization
from molvs.tautomer import TAUTOMER_TRANSFORMS, TAUTOMER_SCORES, MAX_TAUTOMERS, TautomerCanonicalizer, TautomerEnumerator, TautomerTransform
from molvs.fragment import LargestFragmentChooser
from molvs.charge import Reionizer, Uncharger
import argparse
import csv
import pandas as pd
import numpy as np


def normalize(smiles):
    #print(smiles)

    # Generate Mol
    mol1 = Chem.MolFromSmiles(smiles)

    # Uncharge
    uncharger = Uncharger()
    mol2 = uncharger.uncharge(mol1)

    # LargestFragmentChooser
    flagmentChooser = LargestFragmentChooser()

    try:
        mol3 = flagmentChooser(mol2)
    except:
        try:
            mol3 = flagmentChooser(mol1)
        except:
            mol3 = mol1


    # Sanitaize
    Chem.SanitizeMol(mol3)

    # Normalize
    normalizer = Normalizer()
    mol4 = normalizer.normalize(mol3)

    #tautomerCanonicalizer = TautomerCanonicalizer()
    #mol = tautomerCanonicalizer.canonicalize(mol)

    return Chem.MolToSmiles(mol4)


def get_group(groups_node, ns):
    ret = ""
    if groups_node:
        for i, child in enumerate(groups_node.iter(f"{ns}group")):
            if i > 0 :
                ret += ","
            ret += child.text

    return ret


def get_id(drug_node, ns):

    for i, child in enumerate(drug_node.iter(f"{ns}drugbank-id")):
        for attr in child.attrib:
            if attr == "primary":
                return child.text

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_xml", type=str, required=True)
    parser.add_argument("-input_sdf", type=str, required=True)
    parser.add_argument("-output", type=str, required=True)
    args = parser.parse_args()

    name_dict = {}
    smiles_dict = {}

    sdf_sup = Chem.SDMolSupplier(args.input_sdf)
    datas = []
    for i, mol in enumerate(sdf_sup):
        #if i > 1:
        #    break
        if not mol:
            continue
        if mol.HasProp("DRUGBANK_ID"):
            id = mol.GetProp("DRUGBANK_ID")
            if mol.HasProp("COMMON_NAME"):
                name = mol.GetProp("COMMON_NAME")
            smiles = Chem.MolToSmiles(mol)
            new_smiles = normalize(smiles)
            name_dict[id] = name
            smiles_dict[id] = new_smiles
            print(f"{i} {id} {name} {new_smiles}")
            #datas.append([id, name, smiles, new_smiles])

    # df = pd.DataFrame(datas, columns=["ID", "org_smiles", "smiles"])
    # df.set_index("ID", inplace=True, drop=True)
    # df.to_csv(args.output)

    tree = ET.parse(args.input_xml)
    root = tree.getroot()

    ns = "{http://www.drugbank.ca}"
    ids = []
    datas = []
    for i, drug in enumerate(root.iter(f"{ns}drug")):
        id = get_id(drug, ns)
        category = get_group(drug.find(f"{ns}groups"), ns)
        if id and id in smiles_dict:
            print(f"{i}, {id}, {category}")
            ids.append(id)
            datas.append([name_dict[id], category, smiles_dict[id]])

    df = pd.DataFrame(datas, index=ids, columns=[["name", "status", "smiles"]])
    #df.index.name = "ID"
    df.to_csv(args.output)


if __name__ == "__main__":
    main()
