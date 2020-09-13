from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
import csv
import pandas as pd
import numpy as np
import gzip
import _pickle as cPickle
from molvs.normalize import Normalizer, Normalization
from molvs.tautomer import TAUTOMER_TRANSFORMS, TAUTOMER_SCORES, MAX_TAUTOMERS, TautomerCanonicalizer, TautomerEnumerator, TautomerTransform
from molvs.fragment import LargestFragmentChooser
from molvs.charge import Reionizer, Uncharger

def normalize(smiles):
    from copy import deepcopy
    #print(smiles)

    # Generate Mol
    mol = Chem.MolFromSmiles(smiles)

    # Uncharge
    uncharger = Uncharger()
    try:
        mol2 = uncharger.uncharge(mol)
    except:
        mol2 = mol
    mol2 = mol

    # LargestFragmentChooser
    flagmentChooser = LargestFragmentChooser()
    try:
        mol3 = flagmentChooser(mol2)
    except:
        mol3 = mol2

    # Sanitaize
    mol3_tmp = deepcopy(mol3)
    try:
        ret = Chem.SanitizeMol(mol3, catchErrors=False)
    except:
        mol3 = mol3_tmp

    # Normalize
    normalizer = Normalizer()
    try:
        mol4 = normalizer.normalize(mol3)
    except:
        mol4 = mol3

    #print(mol4)
    tautomerCanonicalizer = TautomerCanonicalizer()
    mol4 = tautomerCanonicalizer.canonicalize(mol4)

    new_smiles = Chem.MolToSmiles(mol4)
    #print(new_smiles)
    return new_smiles

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_train", type=str, required=True)
    parser.add_argument("-input_predict", type=str, required=True)
    parser.add_argument("-input_model", type=str, required=True)
    parser.add_argument("-output_csv", type=str, required=True)
    parser.add_argument("-output_report", type=str, required=True)
    args = parser.parse_args()

    #学習データ読み込み
    trains = list()
    with open(args.input_train, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            smiles = row["canonical_smiles"]
            new_smiles = normalize(smiles)
            trains.append((row["id"], new_smiles))


    #予測データの読み込み
    smiles_list = list()
    datas = []
    datas_other = []

    with open(args.input_predict, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(row["id"])
            smiles = row["smiles"]
            new_smiles = normalize(smiles)
            dup_pred = None
            dup_train = None

            if new_smiles is None or Chem.MolFromSmiles(new_smiles) is None:
                print(f"error! {id}")
                new_smiles = smiles

            existFlag = False
            for db_id, tmp_smiles in smiles_list:
                if new_smiles == tmp_smiles:
                    dup_pred = db_id
                    print(f"{id}: same structure exist in predict data! - {db_id}, {tmp_smiles}")
                    break

            for db_id, tmp_smiles in trains:
                if new_smiles == tmp_smiles:
                    dup_train = db_id
                    print(f"{id}: same structure exist in train data! - {db_id}, {tmp_smiles}")
                    break

            smiles_list.append((row["id"], new_smiles))
            mol = Chem.MolFromSmiles(new_smiles)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048, useFeatures=False, useChirality=False)
            fp = pd.Series(np.asarray(fp)).values
            data = []
            data.append(row["id"])
            data.extend(fp)
            datas.append(data)
            datas_other.append((row["id"], row["name"], row["status"], dup_pred, dup_train))

        columns = list()
        columns.append("id")
        columns.extend(["Bit_" + str(i+1) for i in range(2048)])
        df = pd.DataFrame(data=datas, columns=columns)
        df.set_index("id", inplace=True, drop=True)

        df_other = pd.DataFrame(data=datas_other, columns=["id", "name", "status", "dup_predict", "dup_train"])
        df_other.set_index("id", inplace=True, drop=True)

        #一旦保存
        df.to_csv(args.output_csv)

    #df = pd.read_csv(args.input, index_col=0)
    X_vs = df.iloc[:, 0:]

    with gzip.GzipFile(args.input_model, 'r') as f:
        model = cPickle.load(f)

    ad_threshold = 0.70

    y_pred = model.predict(X_vs)
    confidence = model.predict_proba(X_vs)
    confidence = np.amax(confidence, axis=1).round(2)
    ad = confidence >= ad_threshold

    pred = pd.DataFrame({'Prediction': y_pred, 'AD': ad, 'Confidence': confidence}, index=None)
    pred.AD[pred.AD == False] = np.nan
    pred.AD[pred.AD == True] = pred.Prediction.astype(int)

    pred_ad = pred.dropna().astype(int)
    coverage_ad = len(pred_ad) * 100 / len(pred)

    print('VS pred: %s' % pred.Prediction)
    print('VS pred AD: %s' % pred_ad.Prediction)
    print('Coverage of AD: %.2f%%' % coverage_ad)

    pred.index = X_vs.index
    predictions = pd.concat([df_other, pred], axis=1)

    for col in ['Prediction', 'AD']:
        predictions[col].replace(0, 'Inactive', inplace=True)
        predictions[col].replace(1, 'Active', inplace=True)
    print(predictions.head())

    predictions.to_csv(args.output_report)


if __name__ == "__main__":
    main()
