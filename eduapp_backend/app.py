from django.db import connection

def create_table():
    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS videoData (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                age INTEGER
            )
        """)

def insert_data(name, age):
    with connection.cursor() as cursor:
        cursor.execute("INSERT INTO my_table (name, age) VALUES (?, ?)", (name, age))

def get_data():
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM my_table")
        return cursor.fetchall()