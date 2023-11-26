import sqlite3

DB_NAME = "data.sqlite"


def connect_to_db():
    return sqlite3.connect(DB_NAME)


def create_tables():
    conn = connect_to_db()
    cur = conn.cursor()

    try:
        # Create stock_data table
        cur.execute('''
        CREATE TABLE IF NOT EXISTS stock_data (
            date TEXT PRIMARY KEY,
            Open FLOAT,
            High FLOAT,
            Low FLOAT,
            "Close" FLOAT,
            "Adj Close" FLOAT,
            Volume INT
        )
        ''')

        # Create sentiment_data table
        cur.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_data (
                date TEXT PRIMARY KEY,
                sentiment_score REAL
            )
        ''')

        # Create another sentiment_data table
        cur.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_data_year (
                date TEXT PRIMARY KEY,
                sentiment_score REAL
            )
        ''')

        conn.commit()

    except sqlite3.Error as e:
        print(f"An error occurred while creating tables: {e}")

    finally:
        conn.close()


def setup_db():
    create_tables()

def list_tables():
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()
    for table in tables:
        print(table[0])
    conn.close()


if __name__ == "__main__":
    setup_db()
