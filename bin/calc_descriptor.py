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
    mol = Chem.MolFromSmiles(smiles)

    # Uncharge
    uncharger = Uncharger()
    mol = uncharger.uncharge(mol)

    # LargestFragmentChooser
    flagmentChooser = LargestFragmentChooser()
    mol = flagmentChooser(mol)

    # Sanitaize
    Chem.SanitizeMol(mol)

    # Normalize
    normalizer = Normalizer()
    mol = normalizer.normalize(mol)

    tautomerCanonicalizer = TautomerCanonicalizer()
    mol = tautomerCanonicalizer.canonicalize(mol)

    return Chem.MolToSmiles(mol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", type=str, required=True)
    parser.add_argument("-output", type=str, required=True)

    args = parser.parse_args()

    #学習データの読み込み
    smiles_list = list()
    datas = []

    with open(args.input, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            org_smiles = row["canonical_smiles"]
            new_smiles = normalize(org_smiles)
            if org_smiles != new_smiles:
                print(org_smiles)
                print(new_smiles)
            else:
                print("same")

            existFlag = False
            for tmp_smiles in smiles_list:
                if new_smiles == tmp_smiles:
                    print("exist!")
                    existFlag = True
                    break

            if not existFlag:

                smiles_list.append(new_smiles)
                mol = Chem.MolFromSmiles(new_smiles)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048, useFeatures=False, useChirality=False)
                fp = pd.Series(np.asarray(fp)).values

                data = []
                data.append(row["id"])
                data.append(int(row["outcome"]))
                data.extend(fp)
                #print(data)
                datas.append(data)

        columns = list()
        columns.append("id")
        columns.append("outcome")
        columns.extend(["Bit_" + str(i + 1) for i in range(2048)])
        df = pd.DataFrame(data=datas, columns=columns)
        df.set_index("id", inplace=True, drop=True)

        #一旦保存
        df.to_csv(args.output)

    #create model




if __name__ == "__main__":
    main()
