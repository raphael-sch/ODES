from time import sleep
import time

from app import get_db_outside_context, update_worker_status

CHECK_FOR_TIMEOUT_EVERY = 25 # 25
GRACE_PERIOD = 70 # 70

def observe_heartbeat():
    while True:
        db = get_db_outside_context()
        _observe_heartbeat(db)
        db.commit()
        db.close()
        sleep(CHECK_FOR_TIMEOUT_EVERY)

def _observe_heartbeat(db):
    timestamp = int(time.time())
    cursor = db.execute(
        'SELECT id, last_seen FROM worker WHERE status != ? AND ? - last_seen > ?',
        ('timeout', timestamp, GRACE_PERIOD)
    )
    for worker in cursor.fetchall():
        worker_id, last_seen = worker
        update_worker_status(db, worker_id, 'timeout')
        db.execute('UPDATE job SET status = ? WHERE worker_id = ? AND status = ?', ('pending', worker_id, 'running'))
        print(f"Worker ({worker_id}) was not seen for {timestamp - last_seen} seconds and set to timeout.")



if __name__ == "__main__":
    observe_heartbeat()
