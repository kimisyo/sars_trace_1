import argparse
import csv
from collections import defaultdict


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-input", type=str, required=True)
    parser.add_argument("-output", type=str, required=True)
    args = parser.parse_args()

    rows_by_id = defaultdict(list)

    # csvを読み込み、CHEMBLID毎にrowをまとめる
    with open(args.input, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_id[row["compound_chembl_id"]].append(row)

    # activeとみなす値
    threshold = 10000

    # CHEMBLID毎に値を処理しながら出力する
    with open(args.output, "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["chembl_id", "canonical_smiles", "value", "outcome"])

        # CHEMBLID毎に値を処理
        for id in rows_by_id:
            # 合計を求める
            total = 0.0
            for row in rows_by_id[id]:
                value = row["standard_value"]
                total += float(value)

            mean = total/len(rows_by_id[id])
            print(f'{id},{mean}')
            outcome = 0
            if mean < threshold:
                outcome = 1
            writer.writerow([id, rows_by_id[id][0]["canonical_smiles"], mean, outcome])


if __name__ == "__main__":
    main()