import os
from gevent import monkey
monkey.patch_all()
from flask import Flask, jsonify, render_template
from datetime import datetime
from flask_socketio import SocketIO, send, emit, join_room, leave_room

from app import get_db, qdb, unpack_data, check_existence
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", 'secret!')
REDIS_URL = os.environ.get("REDIS_URL", 'redis://localhost:6379')
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='gevent',
    message_queue=REDIS_URL
)


##################### SocketIO #########################
# Handle client joining a project
@socketio.on("join_project")
def handle_join_project(data):
    project_id = data.get("project_id")
    if not project_id:
        return
    room = f"project_{project_id}"
    join_room(room)
    print(f"Client joined room {room}")
    #socketio.emit("project_message", {"message": f"Welcome to project {project_id}"}, to=room)

@socketio.on('join')
def on_join(data):
    username = data['username']
    room = data['room']
    join_room(room)
    send(username + ' has entered the room.', to=room)

@socketio.on('leave_project')
def on_leave(data):
    username = data['username']
    room = data['room']
    leave_room(room)
    send(username + ' has left the room.', to=room)

# Broadcast a message to all clients in a project
def broadcast_to_project(project_id, message):
    room = f"project_{project_id}"
    socketio.emit("project_message", {"message": message}, to=room)
    print(f"Broadcasted to {room}: {message}")



##################### GUI #########################
@app.route('/', methods=['GET'])
def view_projects():
    """Render list of projects as HTML using Jinja2 template"""
    db = get_db()
    cursor = db.execute('SELECT id, name, timestamp FROM project ORDER BY timestamp DESC')
    projects_raw = cursor.fetchall()
    db.close()

    # Format timestamps for display
    projects = []
    for project in projects_raw:
        projects.append({
            'id': project['id'],
            'name': project['name'],
            'time_created': datetime.fromtimestamp(project['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        })

    return render_template('projects.html', projects=projects)


@app.route('/project/<int:project_id>', methods=['GET'])
def view_project(project_id):
    """View a single project as HTML with metadata, active workers, and task history"""
    # Get project details
    db = get_db()

    project = qdb("project", "*", project_id)
    # Parse project data
    project_data = {
        'id': project['id'],
        'name': project['name'],
        'dataset': project['dataset'],
        'model': project['model'],
        'hyperparameters': project['hyperparameters'],
        'timestamp': datetime.fromtimestamp(project['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
    }

    metrics_data = dict(kl=[], dev_accuracy=[], train_reward=[])
    cursor = db.execute('SELECT name, value FROM metric WHERE project_id = ? ORDER BY id', (project_id, ))
    for name, value in cursor.fetchall():
        if name not in metrics_data:
            metrics_data[name] = list()
        metrics_data[name].append(value)

    workers = fetch_project_workers(project_id)
    tasks = fetch_project_tasks(project_id)

    return render_template('project_detail.html',
                           project=project_data,
                           metrics=metrics_data,
                           workers=workers,
                           tasks=tasks)


def fetch_project_workers(project_id, max_timeout=3):
    db = get_db()
    cursor = db.execute('SELECT * FROM worker WHERE project_id = ?', (project_id,))
    workers_raw = cursor.fetchall()

    workers_timeout = list()
    for w in workers_raw:
        if w['status'] == 'timeout':
            workers_timeout.append(w)
    workers_timeout.sort(key=lambda w: -w['num_jobs_completed'])
    workers_timeout_keep = set(w['id'] for w in workers_timeout[:max_timeout])

    workers = []
    for w in workers_raw:
        if w['status'] == 'timeout' and w['id'] not in workers_timeout_keep:
            continue
        workers.append({
            'id': w["id"],
            'name': w['name'],
            'country': w['country'],
            'status': w['status'],
            'num_jobs_completed': w['num_jobs_completed'],
            'tokens_per_second': w['tokens_per_second'],
            'last_seen': w['last_seen']
        })
    return workers

@app.route('/api/projects/<int:project_id>/workers', methods=['GET'])
def get_project_workers(project_id):
    workers = fetch_project_workers(project_id)
    return jsonify(workers)


def fetch_project_tasks(project_id):
    db = get_db()
    cursor = db.execute('SELECT * FROM task WHERE project_id = ? ORDER BY id DESC', (project_id,))
    tasks_raw = cursor.fetchall()

    tasks = []
    for t in tasks_raw:
        tasks.append({
            'id': t["id"],
            'parent_id': t["parent_id"],
            'status': t['status'],
            'num_jobs': t['num_jobs'],
            'num_jobs_completed': t['num_jobs_completed'],
            'instance_ids': t['instance_ids'],
            'timestamp_start': t['timestamp_start'],
            'timestamp_finish': t['timestamp_finish']
        })
    tasks.sort(key=lambda t: -t['timestamp_start'])
    return tasks[:50]

@app.route('/api/projects/<int:project_id>/tasks', methods=['GET'])
def get_project_tasks(project_id):
    tasks = fetch_project_tasks(project_id)
    return jsonify(tasks)

@app.route('/api/projects/<int:project_id>/metrics/<metric_name>', methods=['GET'])
def get_metrics(project_id, metric_name):
    db = get_db()
    cursor = db.execute('SELECT value FROM metric WHERE project_id = ? AND name = ?', (project_id, metric_name))
    metric = cursor.fetchall()
    values = [float(m['value']) for m in metric]
    return jsonify(values)

@app.route('/projects', methods=['GET'])
def get_projects():
    """View all projects"""
    db = get_db()
    cursor = db.execute('SELECT id, name, dataset FROM project')
    projects = [dict(row) for row in cursor.fetchall()]
    return jsonify(projects)


@app.route('/projects/get/<int:project_id>', methods=['GET'])
def get_project(project_id):
    """View a single project"""
    project = qdb("project", "*", project_id)
    return jsonify(dict(project))

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
