"""
Sets up the SQLite user database from data/seed_data.json.
PII lives only in seed_data.json which is git-ignored.
Copy data/seed_data.example.json → data/seed_data.json and fill in your values.
"""
import json
import os
import sqlite3


def setup_database(seed_path: str = "data/seed_data.json", db_path: str = "data/user_data.db") -> None:
    if not os.path.exists(seed_path):
        raise FileNotFoundError(
            f"Seed data not found at {seed_path}. "
            "Copy data/seed_data.example.json to data/seed_data.json and fill in your values."
        )

    with open(seed_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            full_name TEXT, age INTEGER, gender TEXT, date_of_birth TEXT,
            place_of_birth TEXT, mother_name TEXT, mother_birth_year INTEGER,
            father_name TEXT, father_birth_year INTEGER, sibling_count INTEGER,
            sibling_name TEXT, sibling_birth_year INTEGER, sibling_gender TEXT,
            marital_status TEXT, email TEXT, phone_number TEXT, city TEXT,
            hometown TEXT, languages_spoken TEXT, favorite_cuisines TEXT,
            shoe_size INTEGER, height REAL, weight REAL, eye_color TEXT,
            hair_color TEXT, hobbies TEXT
        )
    """)

    cursor.execute("SELECT COUNT(*) FROM users WHERE full_name = ?", (data["full_name"],))
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO users (
                full_name, age, gender, date_of_birth, place_of_birth,
                mother_name, mother_birth_year, father_name, father_birth_year,
                sibling_count, sibling_name, sibling_birth_year, sibling_gender,
                marital_status, email, phone_number, city, hometown,
                languages_spoken, favorite_cuisines, shoe_size, height, weight,
                eye_color, hair_color, hobbies
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["full_name"], data["age"], data["gender"], data["date_of_birth"],
            data["place_of_birth"], data["mother_name"], data["mother_birth_year"],
            data["father_name"], data["father_birth_year"], data["sibling_count"],
            data.get("sibling_name", ""), data.get("sibling_birth_year"),
            data.get("sibling_gender", ""), data["marital_status"],
            data["email"], data["phone_number"], data["city"], data["hometown"],
            data["languages_spoken"], data["favorite_cuisines"], data["shoe_size"],
            data["height"], data["weight"], data["eye_color"], data["hair_color"],
            data["hobbies"],
        ))
        conn.commit()
        print(f"Inserted record for {data['full_name']}.")
    else:
        print(f"Record for {data['full_name']} already exists. Skipping insert.")

    cursor.close()
    conn.close()
    print("Database setup completed successfully!")


if __name__ == "__main__":
    setup_database()
