import json

with open('../../data/test_conversations_single_qa.json') as f:
    single_qa = json.load(f)

with open('../../data/eval_single_qa.json', 'w') as f:
    for row in single_qa:
        line = json.dumps({
            "question_id": row["id"],
            "image": row["image"],
            "text": row["conversations"][0]["value"].replace('<image>\n', ''),
            "category": "generic"
        })
        f.write(line + '\n')

with open('../../data/test_conversations.json') as f:
    conv = json.load(f)

with open('../../data/eval.json', 'w') as f:
    for row in conv:
        for i in range(0, len(row["conversations"]), 2):
            line = json.dumps({
                "question_id": row["id"],
                "image": row["image"],
                "text": row["conversations"][i]["value"].replace('<image>\n', ''),
                "category": "generic"
            })
            f.write(line + '\n')
