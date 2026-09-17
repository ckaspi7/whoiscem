"""
Sets up the SQLite user database from data/seed_data.json.

The schema deliberately holds no contact details, no date of birth and no family
information: this database is read straight into model context, so anything it
stores is something a language model may repeat. Copy
data/seed_data.example.json to data/seed_data.json (git-ignored) and fill it in.
"""
import json
import os
import sqlite3

# Mirrors tools/personal_tool._SAFE_FIELDS. Adding a column here means agreeing
# that a model may say it out loud.
COLUMNS = (
    "full_name",
    "gender",
    "birth_year",
    "place_of_birth",
    "hometown",
    "city",
    "marital_status",
    "languages_spoken",
    "favorite_cuisines",
    "hobbies",
    "eye_color",
    "hair_color",
)

# Columns from the previous schema that must never come back.
RETIRED_COLUMNS = (
    "age", "date_of_birth", "email", "phone_number",
    "mother_name", "mother_birth_year", "father_name", "father_birth_year",
    "sibling_count", "sibling_name", "sibling_birth_year", "sibling_gender",
    "shoe_size", "height", "weight",
)


def setup_database(seed_path: str = "data/seed_data.json", db_path: str = "data/user_data.db") -> None:
    if not os.path.exists(seed_path):
        raise FileNotFoundError(
            f"Seed data not found at {seed_path}. "
            "Copy data/seed_data.example.json to data/seed_data.json and fill in your values."
        )

    with open(seed_path, encoding="utf-8") as f:
        data = json.load(f)

    ignored = sorted(set(data) & set(RETIRED_COLUMNS))
    if ignored:
        print(f"Ignoring sensitive fields present in {seed_path}: {', '.join(ignored)}")

    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")

    cursor.execute(
        "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, "
        + ", ".join(f"{c} {'INTEGER' if c == 'birth_year' else 'TEXT'}" for c in COLUMNS)
        + ")"
    )

    existing = {row[1] for row in cursor.execute("PRAGMA table_info(users)")}
    stale = sorted(existing & set(RETIRED_COLUMNS))
    if stale:
        print(
            f"WARNING: {db_path} predates the slimmed schema and still stores: {', '.join(stale)}.\n"
            "         The tool will not read them, but delete the file and re-run to drop them."
        )

    writable = [c for c in COLUMNS if c in existing]
    cursor.execute("SELECT COUNT(*) FROM users WHERE full_name = ?", (data["full_name"],))
    if cursor.fetchone()[0] == 0:
        placeholders = ", ".join("?" for _ in writable)
        cursor.execute(
            f"INSERT INTO users ({', '.join(writable)}) VALUES ({placeholders})",
            tuple(data.get(c) for c in writable),
        )
        conn.commit()
        print(f"Inserted record for {data['full_name']}.")
    else:
        print(f"Record for {data['full_name']} already exists. Skipping insert.")

    cursor.close()
    conn.close()
    print("Database setup completed successfully!")


if __name__ == "__main__":
    setup_database()
