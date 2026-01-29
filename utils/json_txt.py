import json

with open("/data/yujk/hovernet2feature/HEST-Bench/HCC/var_50genes.json", "r") as f:
    data = json.load(f)

genes = data["genes"]

with open("/data/yujk/hovernet2feature/HEST-Bench/HCC/var_50genes.txt", "w") as f:
    for gene in genes:
        f.write(gene + "\n")
