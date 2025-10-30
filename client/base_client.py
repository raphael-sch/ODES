import threading
import time
import requests
from time import sleep
from urllib.parse import urljoin

HEARTBEAT_EVERY = 20
HEARTBEAT_PATIENCE = 30
JOB_FETCH_COOLDOWN = 5
JOB_DELIVER_COOLDOWN = 10
WORKER_REGISTER_COOLDOWN = 20

VERSION = '0.001'

#def main():
#    while True:
#        job = fetch_job()
#        results = execute_job()
#        deliver_job_results()


def execute_job(job):
    raise NotImplementedError()

def deliver_job_results(base_url, worker_id, job, results):
    url = urljoin(base_url, "/jobs/deliver")
    data = dict(job_id=job['job_id'],
                worker_id=worker_id,
                results=results)

    try:
        response = requests.post(url, json=data)
        result = response.json()
    except Exception as e:
        print(f"Error while fetching job: {e}")
        sleep(JOB_DELIVER_COOLDOWN)
        return deliver_job_results(base_url, worker_id, job, results)

    status = result['status']
    if status == 'failed':
        msg = result['msg']
        print(f"Failed to deliver job: {msg}")
        return -1
    return 0


def fetch_job(base_url, project_id, worker_id):
    url = urljoin(base_url, "/jobs/fetch")
    data = dict(project_id=project_id, worker_id=worker_id)
    job = None
    while job is None:
        try:
            response = requests.post(url, json=data)
            result = response.json()
        except Exception as e:
            print(f"Error while fetching job: {e}")
            sleep(JOB_FETCH_COOLDOWN)
            continue
        status = result['status']
        if status != 'success':
            msg = result['msg']
            print(f"Failed to fetch job: {msg}")
            sleep(JOB_FETCH_COOLDOWN)
            continue
        job = result['job']
        print(f"Got new job with id: {job['job_id']}")
    return job


def startup(base_url, project_id, worker_name):
    worker_id, model_name, test_job = register_worker(base_url, project_id, worker_name)

    hb_thread = threading.Thread(target=heartbeat, args=(base_url, worker_id, project_id), daemon=True)
    hb_thread.start()
    return worker_id, model_name, test_job

def register_worker(base_url, project_id, name):
    url = urljoin(base_url, "/workers/register")
    data = dict(project_id=project_id, name=name, version=VERSION)
    try:
        response = requests.post(url, json=data)
        result = response.json()
    except Exception as e:
        print(f"Error while registering worker. Please restart script: {e}")
        sleep(WORKER_REGISTER_COOLDOWN)
        return register_worker(base_url, project_id, name)

    status = result['status']
    if status != 'success':
        msg = result['msg']
        print(f"Failed to register worker: {msg}")
        sleep(WORKER_REGISTER_COOLDOWN)
        return register_worker(base_url, project_id, name)

    worker_id = result['worker_id']
    model_name = result['model_name']
    test_job = result['test_job']
    print('Registered worker and got id:', worker_id)
    return worker_id, model_name, test_job

def heartbeat(base_url, worker_id, project_id):
    """Send periodic heartbeat requests to a server."""
    url = urljoin(base_url, f"/workers/{project_id}/{worker_id}/heartbeat")
    patience = HEARTBEAT_PATIENCE
    while True:
        if patience == 0:
            exit()
        try:
            requests.get(url)
            patience = HEARTBEAT_PATIENCE
        except Exception as e:
            print(f"Heartbeat error: {e}")
            patience -= 1
        time.sleep(HEARTBEAT_EVERY)


#if __name__ == "__main__":
#    args = parser.parse_args()
#    print('args base', args)

#    startup(args)
#    main(args)
