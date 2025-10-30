import sqlite3
import os

name = 'prod'  # dev, prod
db_file = f'{name}.db'

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
db_path = os.path.join(current_dir, db_file)


def create_db():
    # Check if the database file already exists
    if not os.path.exists(db_path):
        # Read the SQL file
        with open(os.path.join(current_dir, 'schema.sql'), 'r') as file:
            sql_script = file.read()

        # Connect to the SQLite database (will create the file)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA journal_mode=WAL")

        # Execute the SQL script
        cursor.executescript(sql_script)

        # Commit the changes and close the connection
        conn.commit()
        conn.close()

        print(f"{name} Database created successfully!")
    else:
        print(f"{name} Database already exists. No changes made.")

    return db_path


if __name__ == '__main__':
    create_db()