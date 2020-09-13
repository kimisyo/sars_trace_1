import argparse
import csv

import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import rdBase, Chem, DataStructs

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-train", type=str, required=True)
    parser.add_argument("-predict", type=str)
    parser.add_argument("-result", type=str)
    parser.add_argument("-method", type=str, default="PCA", choices=["PCA", "UMAP"])

    args = parser.parse_args()

    # all_train.csvの読み込み, fpの計算
    train_datas = []
    train_datas_active = []
    train_datas_inactive = []

    with open(args.train, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row["canonical_smiles"]

            mol = Chem.MolFromSmiles(smiles)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048, useFeatures=False, useChirality=False)
            train_datas.append(fp)

            if int(row["outcome"]) == 1:
                 train_datas_active.append(fp)
            else:
                 train_datas_inactive.append(fp)

    if args.predict and args.result:
        result_outcomes = []
        result_ads = []

        # 予測結果読み込み
        with open(args.result, "r",encoding="utf-8_sig") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                #print(row)
                if row["Prediction"] == "Active":
                    result_outcomes.append(1)
                else:
                    result_outcomes.append(0)

                result_ads.append(row["Confidence"])

        # drugbank.csvの読み込み, fpの計算
        predict_datas = []
        predict_datas_active = []
        predict_datas_inactive = []
        predict_ads = []
        with open(args.predict, "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                print(i)
                smiles = row["smiles"]
                mol = Chem.MolFromSmiles(smiles)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048, useFeatures=False, useChirality=False)
                predict_datas.append(fp)

                if result_outcomes[i] == 1:
                    predict_datas_active.append(fp)
                else:
                    predict_datas_inactive.append(fp)


    # 分析
    model = None
    if args.method == "PCA":
        model = PCA(n_components=2)
        model.fit(train_datas)

    if args.method == "UMAP":
        model = umap.UMAP()
        model.fit(train_datas)

    result_train = model.transform(train_datas)
    result_train_active = model.transform(train_datas_active)
    result_train_inactive = model.transform(train_datas_inactive)

    plt.title(args.method)
    #plt.scatter(result_train[:, 0], result_train[:, 1], c="blue", alpha=0.1, marker="o")
    plt.scatter(result_train_active[:, 0], result_train_active[:, 1], c="blue", alpha=0.5, marker="o")
    plt.scatter(result_train_inactive[:, 0], result_train_inactive[:, 1], c="blue", alpha=0.5, marker="x")

    # 予測(predict)
    if args.predict and args.result:

        result_predict = model.transform(predict_datas)
        result_predict_active = model.transform(predict_datas_active)
        result_predict_inactive = model.transform(predict_datas_inactive)

        #plt.scatter(result_predict[:, 0], result_predict[:, 1], c=result_ads, alpha=0.1, cmap='viridis_r')
        plt.scatter(result_predict_active[:, 0], result_predict_active[:, 1], c="red", alpha=0.1, marker="o")
        plt.scatter(result_predict_inactive[:, 0], result_predict_inactive[:, 1], c="red", alpha=0.1, marker="x")

    plt.show()


if __name__ == "__main__":
    main()