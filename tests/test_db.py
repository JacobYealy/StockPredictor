import os
import unittest
import sqlite3
import db_setup

DB_NAME = "data.sqlite"

class TestDBSetup(unittest.TestCase):
    """
        The TestDBSetup class provides a suite of unit tests to verify the functionality
        of database setup operations as defined in db_setup.py. The tests ensure that:
            - A database connection can be successfully established.
            - The required tables (stock_data and sentiment_data) can be created successfully.

        Each test initializes a fresh test database and cleans it up after execution, ensuring
        isolation and preventing side effects between test cases.
        """
    def setUp(self):
        # Before each test case, set up a new DB
        db_setup.setup_db()

    def tearDown(self):
        # Cleanup after each test by removing the DB
        try:
            os.remove(DB_NAME)
        except:
            pass

    def test_database_connection(self):
        conn = db_setup.connect_to_db()
        self.assertIsNotNone(conn)
        conn.close()

    def test_create_tables(self):
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()

        # Check if stock_data table exists
        cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='stock_data'")
        self.assertEqual(cur.fetchone()[0], 1, "stock_data table does not exist.")

        # Check if sentiment_data table exists
        cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='sentiment_data'")
        self.assertEqual(cur.fetchone()[0], 1, "sentiment_data table does not exist.")

        conn.close()

if __name__ == "__main__":
    unittest.main()
