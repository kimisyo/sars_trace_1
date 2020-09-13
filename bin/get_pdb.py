import requests
import json
import time
import lxml.html
import argparse
import csv


def get_ligand(ligand_id):
    tmp_url = "https://www.rcsb.org" + ligand_id

    response = requests.get(tmp_url)
    if response.status_code != 200:
        return response.status_code, []

    html = response.text
    root = lxml.html.fromstring(html)
    #print(html)
    print(tmp_url)

    smiles = root.xpath("//tr[@id='chemicalIsomeric']/td[1]/text()")[0]
    inchi = root.xpath("//tr[@id='chemicalInChI']/td[1]/text()")[0]
    inchi_key = root.xpath("//tr[@id='chemicalInChIKey']/td[1]/text()")[0]

    return response.status_code, [smiles, inchi, inchi_key]


def get_structure(structure_id):

    structure_url = "https://www.rcsb.org/structure/"
    tmp_url = structure_url + structure_id
    print(tmp_url)
    html = requests.get(tmp_url).text
    root = lxml.html.fromstring(html)

    macromolecule_trs = root.xpath("//tr[contains(@id,'macromolecule-entityId-')]")
    macromolecule = ""
    for tr in macromolecule_trs:
        print(tr.xpath("@id"))
        macromolecule += tr.xpath("td[position()=1]/text()")[0] + ","
        print(f"macro={macromolecule}")

    binding_trs = root.xpath("//tr[contains(@id,'binding_row')]")

    datas = []
    ids = []

    for tr in binding_trs:

        print(tr.xpath("@id"))
        d1 = tr.xpath("td[position()=1]/a/@href")
        if d1[0] in ids:
            continue

        ids.append(d1[0])
        status_code, values = get_ligand(d1[0])
        ligand_id = d1[0][(d1[0].rfind("/") + 1):]
        print(ligand_id)
        if status_code == 200:
            smiles, inchi, inchi_key = values
            #item = tr.xpath("a/td[position()=2]/text()")[0]
            item = tr.xpath("td[position()=2]/a/text()")[0]
            item = item.strip()
            value = tr.xpath("td[position()=2]/text()")[0]
            value = value.replace(":", "")
            value = value.replace(";", "")
            value = value.replace("&nbsp", "")
            value = value.replace("\n", "")
            print(value)
            values = value.split(" ", 1)
            print(values)
            value = values[0].strip()
            unit = values[1].strip()

            datas.append([ligand_id, smiles, inchi, inchi_key, item, value, unit, macromolecule])

        time.sleep(1)

    return datas


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-output", type=str, required=True)
    args = parser.parse_args()

    base_url = "https://www.rcsb.org/search/data"
    payloads = {"query":{"type":"group","logical_operator":"and","nodes":[{"type":"group","logical_operator":"and","nodes":[{"type":"group","logical_operator":"or","nodes":[{"type":"group","logical_operator":"and","nodes":[{"type":"terminal","service":"text","parameters":{"attribute":"rcsb_binding_affinity.value","negation":False,"operator":"exists"},"node_id":0},{"type":"terminal","service":"text","parameters":{"attribute":"rcsb_binding_affinity.type","operator":"exact_match","value":"IC50"},"node_id":1}],"label":"nested-attribute"},{"type":"group","logical_operator":"and","nodes":[{"type":"terminal","service":"text","parameters":{"attribute":"rcsb_binding_affinity.value","negation":False,"operator":"exists"},"node_id":2},{"type":"terminal","service":"text","parameters":{"attribute":"rcsb_binding_affinity.type","operator":"exact_match","value":"Ki"},"node_id":3}],"label":"nested-attribute"}]},{"type":"group","logical_operator":"and","nodes":[{"type":"terminal","service":"text","parameters":{"operator":"exact_match","negation":False,"value":"Severe acute respiratory syndrome-related coronavirus","attribute":"rcsb_entity_source_organism.taxonomy_lineage.name"},"node_id":4}]}],"label":"text"}],"label":"query-builder"},"return_type":"entry","request_options":{"pager":{"start":0,"rows":100},"scoring_strategy":"combined","sort":[{"sort_by":"score","direction":"desc"}]},"request_info":{"src":"ui","query_id":"e757fdfd5f9fb0efa272769c5966e3f4"}}
    print(json.dumps(payloads))

    response = requests.post(
        base_url,
        json.dumps(payloads),
        headers={'Content-Type': 'application/json'})

    datas = []
    for a in response.json()["result_set"]:
        structure_id = a["identifier"]
        datas.extend(get_structure(structure_id))
        time.sleep(1)

    with open(args.output, "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["ligand_id", "canonical_smiles", "inchi", "inchi_key", "item", "value", "unit"])

        for data in datas:
            writer.writerow(data)


if __name__ == "__main__":
    main()

