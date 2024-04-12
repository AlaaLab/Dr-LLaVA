import json
import os
### create the question and answers json file for test set
# operating on the test data
with open('../../ucsf_data_rl/LLaVA_heme_train_rl.json') as f:
    data = json.load(f)


# create a new list to store the data
new_data_q = []
new_data_a = []
new_data_q_rl = []
new_data_a_rl = []
for i in range(len(data)):
    id = data[i]['id']
    image = data[i]['image']
    conversations = data[i]['conversations']
    diagnosis = data[i]['diagnosis']

    new_question = ''
    new_answer = ''


    for j in range(int(len(conversations)/2)):

        new_data_q.append({'question_id': id, 'image': image, 'text': conversations[2*j]['value'].replace('<image>','').replace('\n','').split('. ')[-1], 'category': diagnosis})

        new_data_a.append({'answer_id': id, 'image': image, 'text': conversations[2*j+1]['value'], 'category': diagnosis})

        new_question = new_question + conversations[2*j]['value'].replace('<image>','').replace('\n','') + ' '
        new_answer = new_answer + conversations[2*j+1]['value'] + ' '
    new_data_q_rl.append({'question_id': id, 'image': image, 'text': new_question, 'category': diagnosis})
    new_data_a_rl.append({'answer_id': id, 'image': image, 'text': new_answer, 'category': diagnosis})



# save the new data
# Open the JSON file in write mode
if os.path.exists('../../ucsf_data_rl/LLaVA_heme_train_q.json'):
    print('The file already exists')
    # overwrite the file
    print('Overwriting the file')
    os.remove('../../ucsf_data_rl/LLaVA_heme_train_q.json')

with open('../../ucsf_data_rl/LLaVA_heme_train_q.json', "w") as json_file:
    # Iterate through the list of dictionaries

    for element in new_data_q:
    # Write each element to the file in JSON format, followed by a newline character
        json_file.write(json.dumps(element) + "\n")

if os.path.exists('../../ucsf_data_rl/LLaVA_heme_train_a.json'):
    print('The file already exists')
    # overwrite the file
    print('Overwriting the file')
    os.remove('../../ucsf_data_rl/LLaVA_heme_train_a.json')

with open('../../ucsf_data_rl/LLaVA_heme_train_a.json', "w") as json_file:
    # Iterate through the list of dictionaries

    for element in new_data_a:
    # Write each element to the file in JSON format, followed by a newline character
        json_file.write(json.dumps(element) + "\n")
