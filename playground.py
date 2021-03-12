data_path = 'data/glue/MNLI/dev_matched.tsv'
with open(data_path) as fin:
    fin.readline()
    line = fin.readline().strip().split('\t')
    for segment in line:
        print(segment)