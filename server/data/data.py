import string
import json
import os

import tqdm
from datasets import load_dataset

LETTERS = string.ascii_uppercase
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)

prompt_mcqa = "Please first think about the reasoning process in the mind and then provide the answer. The reasoning process and answer (letter of correct option) are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> $LETTER </answer>\n\n"
cache = dict()


def get_prompt(name):
    if name == 'aqua':
        return prompt_mcqa
    raise ValueError(f"Dataset {name} not available.")

def get_dataset(name):
    if name in cache:
        return cache[name]
    if name == 'aqua':
        dataset = get_aqua()
    else:
        raise ValueError(f"Dataset {name} not available.")
    cache[name] = dataset
    return cache[name]

def get_aqua():
    cache_file = os.path.join(current_dir, 'cached', 'aqua.json')
    data = get_maybe_cached(cache_file)
    if data is not None:
        return data

    dataset = load_dataset("deepmind/aqua_rat", "raw")
    data = dict()
    for split in ['train', 'validation']:
        data[split] = list()
        for row in tqdm.tqdm(dataset[split], desc=f'Preprocess AQuA {split}'):
            question = row['question'].strip()
            options_raw = row['options']
            answer = row['correct']
            options_text = ""
            for option, letter in zip(options_raw, LETTERS):
                assert option.startswith(letter + ')')
                option = letter + ')' + option[2:].strip()
                options_text += option + '\n'
            text = 'Question: ' + question + '\n' + options_text
            data[split].append((len(data[split]), text, answer))
    data['dev'] = data['validation']
    del data['validation']
    with open(cache_file, 'w') as f:
        json.dump(data, f, indent=2)
    return data


def get_maybe_cached(cache_file):
    if os.path.isfile(cache_file):
        with open(cache_file) as f:
            data = json.load(f)
        return data
    return None


def extract_answers(results_raw):
    answers = list()
    for answer_raw in results_raw:
        if answer_raw is None:
            answers.append(None)
        else:
            answer_raw = answer_raw.strip()
            answer = ''.join([c for c in answer_raw if c.isupper()])
            if len(answer) == 0:
                answers.append('')
            else:
                answers.append(answer[0])
    return answers


def get_reward(answers, oracle_answers):
    reward = 0.0
    num_correct = 0
    assert len(answers) == len(oracle_answers)
    for answer, oracle in zip(answers, oracle_answers):
        if answer is None:
            continue
        if len(answer) == 0: # has answer tags
            reward += 0.1
        if answer == oracle: # correct answer letter
            reward += 1.0
            num_correct += 1
        elif len(answer) == 1: # incorrect answer letter
            reward += 0.2

    return reward / len(answers), num_correct


if __name__ == '__main__':
    get_aqua()
