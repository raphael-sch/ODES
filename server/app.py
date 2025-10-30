from gevent import monkey
monkey.patch_all()
import json
import random
import requests
import os

from flask import Flask, request, jsonify, g, abort
from flask_socketio import SocketIO
import sqlite3
import time
import numpy as np
from threading import Thread

from data.data import get_dataset, extract_answers, get_reward, get_prompt
from db.create_db import create_db
app = Flask(__name__)
REDIS_URL = os.environ.get("REDIS_URL", 'redis://localhost:6379')
socketio = SocketIO(message_queue=REDIS_URL)

DATABASE = create_db()
MIN_CLIENT_VERSION = 0.001


##################### ORCHESTRATION #########################

@app.route('/projects/create', methods=['POST'])
def create_project():
    """Create a new project"""
    timestamp = int(time.time())
    data = request.get_json()
    name = unpack_data(["name"], data)

    dataset = data.get('dataset', 'aqua')
    model_name = data.get('model', 'Qwen/Qwen2.5-0.5B-Instruct')

    hyperparameters = data.get('hyperparameters', dict())
    hyperparameters['population'] = hyperparameters.get('population', 32)
    hyperparameters['alpha'] = hyperparameters.get('alpha', 0.0005)
    hyperparameters['sigma'] = hyperparameters.get('sigma', 0.001)
    hyperparameters['batch_size'] = hyperparameters.get('batch_size', 16)
    hyperparameters['temperature'] = hyperparameters.get('temperature', 0.0)
    hyperparameters['max_tokens'] = hyperparameters.get('max_tokens', 2048)
    hyperparameters['eval_steps'] = hyperparameters.get('eval_steps', 3)
    hyperparameters['max_population'] = hyperparameters.get('max_population', 1024)

    db = get_db()
    cursor = db.execute(
        'INSERT INTO project (name, dataset, model, timestamp, hyperparameters) VALUES (?, ?, ?, ?, ?)',
        (name, dataset, model_name, timestamp, json.dumps(hyperparameters))
    )
    db.commit()
    project_id = cursor.lastrowid

    create_new_task(db, parent_task_id=-1, project_id=project_id, split='dev')
    create_new_task(db, parent_task_id=-1, project_id=project_id, split='train')
    db.commit()
    return jsonify({'status': 'success', 'id': project_id})


@app.route('/workers/register', methods=['POST'])
def register_worker():
    """Register a new worker"""
    timestamp = int(time.time())
    name, project_id, version = unpack_data(["name", "project_id", "version"])
    if float(version) < MIN_CLIENT_VERSION:
        return jsonify({'status': 'failed', 'msg': f'Client version too old. Please update to at least {MIN_CLIENT_VERSION}'})

    if request.headers.getlist("X-Forwarded-For"):
        ip = request.headers.getlist("X-Forwarded-For")[0]
    else:
        ip = request.remote_addr
    country_code = get_country_code(ip)

    model_name = qdb("project", ["model"], project_id)
    db = get_db()
    cursor = db.execute(
        'INSERT INTO worker (project_id, last_seen, status, timestamp, name, country) VALUES (?, ?, ?, ?, ?, ?)',
        (project_id, timestamp, 'waiting', timestamp, name, country_code)
    )
    db.commit()
    worker_id = cursor.lastrowid

    data = get_worker_data(worker_id, name, country_code, 'waiting', 0, timestamp)
    socketio.emit("worker_update", data, to=f"project_{project_id}")


    dataset_name, hyperparameters = qdb("project", ["dataset", "hyperparameters"], project_id)
    hyperparameters = json.loads(hyperparameters)
    dataset = get_dataset(dataset_name)
    instance_ids = [eid for eid, _, _ in dataset['dev']][:32]
    instances = get_instances(instance_ids, dataset_name, 'dev')
    data = dict()
    data['instances'] = instances
    data['hyperparameters'] = hyperparameters
    data['prompt'] = get_prompt(dataset_name)


    return jsonify({'status': 'success', 'worker_id': worker_id, 'model_name': model_name, 'test_job': data})

def get_worker_data(worker_id, name, country, status, num_jobs_completed, last_seen):
    worker = {
        'id': worker_id,
        'name': name,
        'country': country,
        'status': status,
        'num_jobs_completed': num_jobs_completed,
        'last_seen': last_seen
    }
    return dict(worker=worker)

@app.route('/workers/<int:project_id>/<int:worker_id>/heartbeat', methods=['GET'])
def worker_heartbeat(project_id, worker_id):
    """Update worker's last_seen timestamp to indicate it's still alive"""
    timestamp = int(time.time())
    db = get_db()
    # Update last_seen timestamp
    db.execute('UPDATE worker SET last_seen = ? WHERE id = ?', (timestamp, worker_id))
    db.commit()
    #emit_worker_update(worker_id, project_id, last_seen=timestamp)
    return jsonify({'status': 'success'})


@app.route('/jobs/fetch', methods=['POST'])
def fetch_job():
    timestamp = int(time.time())
    worker_id, project_id = unpack_data(['worker_id', 'project_id'])
    check_existence(worker_id, 'worker')

    db = get_db()
    # If worker already has jobs, then reset them to pending.
    db.execute("UPDATE job SET status = 'pending' WHERE worker_id = ? AND status = 'running'", (worker_id,))

    # Atomic: update and return in one statement
    cursor = db.execute('''
                        UPDATE job
                        SET status          = 'running',
                            worker_id       = ?,
                            timestamp_start = ?
                        WHERE id = (SELECT id
                                    FROM job
                                    WHERE status = 'pending'
                                      AND project_id = ?
                                    ORDER BY id ASC
                                    LIMIT 1)
                        RETURNING *
                        ''', (worker_id, timestamp, project_id))
    job = cursor.fetchone()

    if job is None:
        return jsonify({'status': 'failed', 'msg': 'No job available.'})

    task_id = job['task_id']
    instance_ids, split, weight_state = qdb("task", ["instance_ids", "split", "weight_state"], task_id)
    instance_ids = json.loads(instance_ids)
    weight_state = json.loads(weight_state)
    dataset_name = qdb("project", ["dataset"], project_id)
    instances = get_instances(instance_ids, dataset_name, split)
    dataset, hyperparameters = qdb("project", ["dataset", "hyperparameters"], project_id)
    hyperparameters = json.loads(hyperparameters)

    data = dict()
    data['instances'] = instances
    data['hyperparameters'] = hyperparameters
    data['prompt'] = get_prompt(dataset)
    data['seed'] = job['seed']
    data['job_id'] = job['id']
    data['weight_state'] = weight_state

    db.execute('UPDATE task SET status = ? WHERE id = ?', ('started', task_id,))
    emit_task_update(task_id, project_id, status='started')
    update_worker_status(db, worker_id, 'running')
    db.commit()

    return jsonify({'status': 'success', 'job': data})


@app.route('/jobs/deliver', methods=['POST'])
def deliver_job_results():
    """Deliver results for a completed job"""
    timestamp = int(time.time())
    job_id, results, worker_id = unpack_data(['job_id', 'results', 'worker_id'])
    check_existence(worker_id, 'worker')
    tokens_per_second = results['tokens_per_second']
    kl = results.get('kl')
    results = results['results']

    # Check if job exists
    task_id, status = qdb("job", ["task_id", "status"], job_id)
    if status in ['canceled', 'complete']:
        return jsonify({'status': 'success', 'msg': f'Could not accept results, job status is: {status}'})

    project_id, num_instances = qdb("task", ["project_id", "num_instances"], task_id)

    db = get_db()
    if len(results) != num_instances:
        db.execute('UPDATE job SET status = ? WHERE id = ?', ('pending', job_id))
        db.commit()
        return jsonify({'status': 'success', 'msg': f"Could not accept results because incorrect length"})

    answers = extract_answers(results)
    # Update job status
    db.execute('UPDATE job SET status = ?, answers = ?, timestamp_finish = ? WHERE id = ?',
        ('complete', json.dumps(answers), timestamp, job_id)
    )

    cursor = db.execute('UPDATE task SET num_jobs_completed = num_jobs_completed + 1 WHERE id = ? RETURNING num_jobs_completed',(task_id, ))
    emit_task_update(task_id, project_id, num_jobs_completed=cursor.fetchone()[0])

    cursor = db.execute('UPDATE worker SET num_jobs_completed = num_jobs_completed + 1, tokens_per_second = ? WHERE id = ? RETURNING num_jobs_completed',
                        (tokens_per_second, worker_id))
    emit_worker_update(worker_id, project_id, last_seen=timestamp, status='waiting', tokens_per_second=tokens_per_second, num_jobs_completed=cursor.fetchone()[0])
    update_worker_status(db, worker_id, 'waiting', emit=False)

    if kl is not None:
        update_metric(db, project_id, 'kl', kl)

    db.commit()

    # Execute in background
    thread = Thread(target=maybe_finish_task, args=(task_id, project_id))
    thread.daemon = True
    thread.start()

    return jsonify({'status': 'success'})


def maybe_finish_task(task_id, project_id):
    db = get_db_outside_context()
    timestamp = int(time.time())

    # atomic operation
    cursor = db.execute('''
                        UPDATE task
                        SET status           = 'complete',
                            timestamp_finish = ?
                        WHERE id = ?
                          AND num_jobs_completed >= num_jobs
                          AND status != 'complete'
                        ''', (timestamp, task_id))
    db.commit()  # Commit the status change now

    # Check if we won the race and should proceed
    if cursor.rowcount == 0:
        # TASK NOT FINISHED
        # Either condition wasn't met, or another process already completed it
        db.close()
        return
    emit_task_update(task_id, project_id, status='complete')

    dataset_name, hyperparameters = qdb("project", ["dataset", "hyperparameters"], project_id, db)
    hyperparameters = json.loads(hyperparameters)

    num_train_tasks_completed, split = finish_task(db, task_id, project_id, dataset_name)
    db.commit()

    if split == 'train':
        if num_train_tasks_completed % hyperparameters['eval_steps'] == 0:
            create_new_task(db, task_id, project_id, dataset_name, hyperparameters, split='dev')
        create_new_task(db, task_id, project_id, dataset_name, hyperparameters, split='train')
        db.commit()
    db.close()


def finish_task(db, task_id, project_id, dataset_name):
    # TASK IS COMPLETED
    timestamp = int(time.time())
    instance_ids, split, weight_state = qdb("task",["instance_ids", "split", "weight_state"], task_id, db)
    instance_ids = json.loads(instance_ids)
    instances = get_instances(instance_ids, dataset_name, split)
    weight_state = json.loads(weight_state)

    cursor = db.execute('SELECT id, status, seed, answers FROM job WHERE task_id = ?', (task_id,))
    results = list()
    rewards = list()
    all_num_correct = 0
    total = 0
    for jobid, status, seed, answers in cursor:
        if status == 'complete':
            answers = json.loads(answers)
            total += len(answers)
            oracle_answers = [instance['answer'] for instance in instances]
            reward, num_correct = get_reward(answers, oracle_answers)
            all_num_correct += num_correct
            rewards.append(reward)
            results.append([seed, reward])
        else:
            db.execute('UPDATE job SET status = ? WHERE id = ?', ('canceled', jobid))

    num_train_tasks_completed = None
    if split == 'train':
        # Normalize rewards
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)
        reward_std = max(reward_std, 1e-8)
        new_state = list()
        for seed, reward in results:
            reward_normed = (reward - reward_mean) / reward_std
            new_state.append([reward_normed, seed])

        weight_state.append(new_state)
        db.execute('UPDATE task SET results = ?, weight_state = ? WHERE id = ?',
                   (json.dumps(results), json.dumps(weight_state), task_id))

        task_reward = sum(rewards) / len(results)
        update_metric(db, project_id, 'train_reward', task_reward)

        cursor = db.execute(
            'UPDATE project SET num_tasks_completed = num_tasks_completed + 1 WHERE id = ? RETURNING num_tasks_completed;',
            (project_id,)
        )
        num_train_tasks_completed = cursor.fetchone()[0]

    if split == 'dev':
        accuracy = all_num_correct / total
        update_metric(db, project_id, 'dev_accuracy', accuracy)

    return num_train_tasks_completed, split

def create_new_task(db, parent_task_id, project_id, dataset_name=None, hyperparameters=None, split='train'):
    if dataset_name is None or hyperparameters is None:
        dataset_name, hyperparameters = qdb("project", ["dataset", "hyperparameters"], project_id, db)
        hyperparameters = json.loads(hyperparameters)
    min_population = hyperparameters['population']
    max_population = hyperparameters['max_population']
    batch_size = hyperparameters['batch_size']

    # get data instances
    dataset = get_dataset(dataset_name)
    if split == 'train':
        instances = random.sample(dataset[split], batch_size)
        instance_ids = [eid for eid, _, _ in instances]
    elif split == 'dev':
        instance_ids = [eid for eid, _, _ in dataset[split]]
    else:
        raise ValueError()

    # get weight state
    weight_state = json.dumps([])
    if parent_task_id != -1:
        weight_state = qdb("task", ["weight_state"], parent_task_id, db)

    # calculate population
    cursor = db.execute('SELECT COUNT(*) FROM worker WHERE project_id = ? AND status != ?', (project_id, 'timeout'))
    num_workers = cursor.fetchone()[0]
    population = max(min_population, min(num_workers * 4, max_population))
    if split == 'dev':
        population = 1

    # create task
    timestamp = int(time.time())
    cursor = db.execute(
        'INSERT INTO task (project_id, status, num_jobs, parent_id, instance_ids, num_instances, split, weight_state, timestamp_start) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (project_id, 'pending', population, parent_task_id, json.dumps(instance_ids), len(instance_ids), split, weight_state, timestamp)
    )
    task_id = cursor.lastrowid
    emit_task_update(task_id, project_id, status='pending', num_jobs=population, num_jobs_completed=0, parent_id=parent_task_id, split=split, timestamp_start=timestamp)

    # create jobs
    for _ in range(population):
        seed = random.randint(1_000, 100_000)
        if split == 'dev':
            seed = -1
        db.execute('INSERT INTO job (task_id, project_id, status, seed) VALUES (?, ?, ?, ?)',
            (task_id, project_id, 'pending', seed)
        )

##################### UPDATES #########################

def update_metric(db, project_id, name, value):
    timestamp = int(time.time())
    db.execute('INSERT INTO metric (project_id, timestamp, name, value) VALUES (?, ?, ?, ?)',
               (project_id, timestamp, name, str(value)))
    emit_metric_update(project_id, name, value)

def update_worker_status(db, worker_id, status, emit=True):
    timestamp = int(time.time())
    cursor = db.execute(
        'UPDATE worker SET last_seen = ?, status = ? WHERE id = ? RETURNING project_id',
        (timestamp, status, worker_id)
    )
    if emit:
        project_id = cursor.fetchone()[0]
        emit_worker_update(worker_id, project_id, last_seen=timestamp, status=status)


##################### socketIO #########################

def emit_worker_update(worker_id, project_id, **kwargs):
    data = dict(worker=dict(id=worker_id, **kwargs))
    socketio.emit("worker_update", data, to=f"project_{project_id}")

def emit_task_update(task_id, project_id, **kwargs):
    data = dict(task=dict(id=task_id, **kwargs))
    socketio.emit("task_update", data, to=f"project_{project_id}")

def emit_metric_update(project_id, name, value):
    data = dict(metric=name, value=value)
    socketio.emit("metrics_update", data, to=f"project_{project_id}")


##################### HELPERS #########################

def get_db():
    if "db" not in g:  # store connection in Flask’s context object
        g.db = sqlite3.connect(DATABASE)
        g.db.execute('PRAGMA foreign_keys = ON')
        g.db.row_factory = sqlite3.Row  # optional: access columns by name
    return g.db

def get_db_outside_context():
    conn = sqlite3.connect(DATABASE)
    conn.execute('PRAGMA foreign_keys = ON')
    conn.row_factory = sqlite3.Row  # optional: access columns by name
    return conn

@app.teardown_appcontext
def close_db(error):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def qdb(table, column_names, object_id, db=None):
    if db is None:
        db = get_db()
    if type(column_names) == str:
        column_names = [column_names]
    column_names_str = ', '.join(column_names)
    cursor = db.execute(f'SELECT {column_names_str} FROM {table} WHERE id = ?', (object_id,))
    result = cursor.fetchone()
    if result is None:
        abort(jsonify({"status": "error", "msg": f"{table} with id {object_id} not found"}))
    if len(column_names) == 1:
        if column_names[0] == "*":
            return result
        return result[column_names[0]]
    return (result[key] for key in column_names)

def check_existence(object_id, table):
    # Check if object_id exists
    db = get_db()
    exists = db.execute(f'SELECT 1 FROM {table} WHERE id = ?', (object_id, )).fetchone()
    if not exists:
        message = f'{table} ID ({object_id}) does not exist.'
        abort(jsonify({'status': 'failed', 'msg': message}))

def get_country_code(ip):
    if str(ip).startswith('129.206') or str(ip).startswith('127.0.0'):
        # https://imgflip.com/i/7ky8s3
        return random.choice(['de', 'ch', 'us'])
    try:
        response = requests.get(f"https://ipwho.is/{ip}")
        data = response.json()
        return data.get("country_code").lower()
    except Exception:
        return 'globe'

def get_instances(instance_ids, dataset_name, split):
    instances = list()
    for example_id in instance_ids:
        example_id, text, answer = get_dataset(dataset_name)[split][example_id]
        instance = dict(example_id=example_id, text=text, answer=answer)
        instances.append(instance)
    return instances

def unpack_data(keys, data=None):
    if data is None:
        data = request.get_json()
    missing_keys = set(keys) - set(data.keys())
    if missing_keys:
        missing_keys = ', '.join(missing_keys)
        abort(jsonify({'status': 'missing_data', 'msg': f'Keys missing: {missing_keys}'}))
    if len(keys) == 1:
        return data[keys[0]]
    return tuple(data[key] for key in keys)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)