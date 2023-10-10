import sqlite3

DB_NAME = "data.sqlite"


def connect_to_db():
    return sqlite3.connect(DB_NAME)


def create_tables():
    conn = connect_to_db()
    cur = conn.cursor()

    # Create stock_data table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS stock_data (
        date TEXT PRIMARY KEY,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        adj_close REAL,
        volume INTEGER
    )
    ''')

    # Create sentiment_data table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS sentiment_data (
        date TEXT PRIMARY KEY,
        sentiment_score REAL
    )
    ''')

    conn.commit()
    conn.close()


def setup_db():
    create_tables()


if __name__ == "__main__":
    setup_db()
