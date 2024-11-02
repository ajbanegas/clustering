import sys

DATASETS = ["cancer", "covid19", "diabetes", "malaria", "rheumatoid_arthritis"]

def get_lines(foo):
    fr = open(foo, "r")
    lines = fr.readlines()
    new_list = list(filter(lambda x: "Time" in x, lines))
    return new_list

if __name__ == "__main__":
    infoo = sys.argv[1]
    lines = get_lines(infoo)

    for i, line in enumerate(lines):
        line = line.replace("Time ", "").replace(" sec", "")
        data = line.split(":")

        dataset = DATASETS[i % len(DATASETS)]
        algorithm = data[0].split(" - ")[0]
        database = data[0].split(" - ")[1]
        time = str(round(float(data[1]), 3)).replace(".", ",")

        print(f"{algorithm}\t{database}\t{dataset}\t{time}")
