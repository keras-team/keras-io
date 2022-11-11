import csv
import re

output = {}

abbr_to_full_name = {
    "CV": "Computer Vision",
    "NLP": "Natural Language Processing",
    "RL": "Reinforcement Learning",
    "GNN": "Graph Neural Networks",
}

with open('/usr/local/google/home/chenmoney/Downloads/keras-io.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        name = row[0]
        category_str = row[1]
        url =row[2]

        category_strs = category_str.split(",")
        categories = []
        for s in category_strs:
            real_name = abbr_to_full_name.get(s, s)
            categories.append(real_name)
        url_no_hashtag = url.split("#")

        result = re.search("examples/(.+)$", url_no_hashtag[0])
        path = result.groups()[0]



    print(f'Processed {line_count} lines.')
