import random
import argparse
from time import sleep

import base_client

parser = argparse.ArgumentParser()
parser.add_argument("--project_id", type=int, required=True)
parser.add_argument('--url', default="http://127.0.0.1:5000", type=str)
args = parser.parse_args()

print('args mock client', args)
EXECUTE_TIME = 5

def main():
    while True:
        job = base_client.fetch_job(args.url, args.project_id, args.worker_id)
        results = execute_job(job)
        base_client.deliver_job_results(args.url, args.worker_id, job, results)


def execute_job(job):
    job_id = job['job_id']
    instances = job['instances']
    seed = job['seed']
    questions = [instance['text'] for instance in instances]
    sleep(EXECUTE_TIME)
    results = [random.choice(['A', 'B', 'C', 'D', ' ', None]) for _ in questions]
    print(f"job {job_id} executed")
    results = dict(results=results, tokens_per_second=random.randint(1337, 9001))
    if seed == -1:
        results['kl'] = random.random()
    return results


if __name__ == "__main__":
    worker_id, model_name, test_job = base_client.startup(args.url, args.project_id, 'H100')
    args.worker_id = worker_id
    main()
