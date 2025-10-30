CREATE TABLE project (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT,
    dataset   TEXT NOT NULL,
    model   TEXT NOT NULL,
    num_tasks_completed   INT DEFAULT 0,
    timestamp INT,
    hyperparameters      TEXT
);

CREATE TABLE metric (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  INT NOT NULL,
    timestamp INT,
    name      TEXT,
    value   TEXT,
    FOREIGN KEY (project_id) REFERENCES project(id)
);
CREATE INDEX idx_metric_project_name ON metric (project_id, name);

CREATE TABLE worker (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  INT NOT NULL,
    last_seen INT,
    status    TEXT,            -- waiting, running, timeout
    num_jobs_completed   INT DEFAULT 0,
    tokens_per_second INT,
    timestamp INT,
    name      TEXT,
    country   TEXT,
    FOREIGN KEY (project_id) REFERENCES project(id)
);
CREATE INDEX idx_worker_project_id ON worker(project_id);

CREATE TABLE task (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  INT NOT NULL,
    status      TEXT,           -- pending, started, complete
    num_jobs   INT NOT NULL,
    num_jobs_completed   INT DEFAULT 0,
    parent_id   INT,
    instance_ids TEXT,
    num_instances   INT,
    split       TEXT, -- train, dev
    weight_state TEXT,
    results      TEXT,
    timestamp_start INT,
    timestamp_finish INT,
    FOREIGN KEY (project_id) REFERENCES project(id)
);
CREATE INDEX idx_task_project_id ON task(project_id);

CREATE TABLE job (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id   INT  NOT NULL,
    project_id  INT NOT NULL,
    worker_id   INT,
    status    TEXT,           -- pending, running, canceled, complete
    timestamp_start INT,
    timestamp_finish INT,
    seed      INT,
    answers      TEXT,
    FOREIGN KEY (task_id) REFERENCES task(id),
    FOREIGN KEY (project_id) REFERENCES project(id),
    FOREIGN KEY (worker_id) REFERENCES worker(id)
);
CREATE INDEX idx_job_task_id ON job(task_id);
CREATE INDEX idx_job_worker_id ON job(worker_id);