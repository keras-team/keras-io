import csv
import re
import json

output = {}

abbr_to_full_name = {
    "CV": "Computer Vision",
    "NLP": "Natural Language Processing",
    "RL": "Reinforcement Learning",
    "GNN": "Graph Neural Networks",
}

with open('/home/chen/Downloads/keras-io.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        name = row[0]
        category_str = row[1]
        url =row[2]

        category_strs = category_str.split(">")
        categories = []
        for s in category_strs:
            s = s.strip()
            real_name = abbr_to_full_name.get(s, s)
            categories.append(real_name)
        url_no_hashtag = url.split("#")
        result = re.search("examples/(.+)$", url_no_hashtag[0])
        path = result.groups()[0]
        on_landing_page = row[3] == "yes"
        line_count += 1
        output[name] = {
            "category": categories,
            "path": path,
            "on_landing_page": on_landing_page,
        }
        
        print(f'Processed {line_count} lines.')


    json = json.dumps(output, indent=4)

    # open file for writing, "w" 
    f = open("examples_metadata.json","w")
    f.write(json)
    f.close()
