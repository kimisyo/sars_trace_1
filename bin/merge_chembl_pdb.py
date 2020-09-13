import argparse
import csv
from collections import defaultdict
import pandas as pd
from rdkit import Chem


def smiles2smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-input_chembl", type=str, required=True)
    parser.add_argument("-input_pdb", type=str, required=True)
    parser.add_argument("-output", type=str, required=True)
    args = parser.parse_args()

    ids = list()
    org_smiles = list()
    can_smiles = list()
    outcomes = list()

    #chemblデータの読み込み(chembl_id, smiles, 阻害値)
    with open(args.input_chembl, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["chembl_id"])
            org_smiles.append(row["canonical_smiles"])
            can_smiles.append(smiles2smiles(row["canonical_smiles"]))
            outcomes.append(row["outcome"])


    #pdbデータの読み込み(pdb_ligand_id, smiles, 阻害値)
    with open(args.input_pdb, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["ligand_id"])
            org_smiles.append(row["canonical_smiles"])
            can_smiles.append(smiles2smiles(row["canonical_smiles"]))
            outcomes.append(row["outcome"])

    #pandasに変換
    df = pd.DataFrame(data={"id": ids, "original_smiles": org_smiles, "canonical_smiles": can_smiles, "outcome":outcomes})

    #重複チェック
    #df["dup"] = df["canonical_smiles"].duplicated()
    df.to_csv(args.output)

    print(df.describe())

    #出力


if __name__ == "__main__":
    main()