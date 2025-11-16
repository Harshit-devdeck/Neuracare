# fix_database.py
import sqlite3

conn = sqlite3.connect("neuracare.db")
c = conn.cursor()

try:
    c.execute("ALTER TABLE conversations ADD COLUMN sentiment_score REAL")
    c.execute("ALTER TABLE conversations ADD COLUMN burnout_score REAL")
    conn.commit()
    print("✅ Database updated successfully!")
except Exception as e:
    print(f"Column might already exist or error: {e}")

conn.close()
