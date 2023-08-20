import sqlite3

def create_table():
    conn = sqlite3.connect('stock_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS stock_data
                 (date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER)''')
    conn.commit()
    conn.close()

def insert_data(data):
    conn = sqlite3.connect('stock_data.db')
    c = conn.cursor()
    c.executemany('INSERT INTO stock_data VALUES (?,?,?,?,?,?)', data)
    conn.commit()
    conn.close()
