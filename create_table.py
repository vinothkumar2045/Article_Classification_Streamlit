import psycopg2

conn = psycopg2.connect(
    host="YOUR_RDS_ENDPOINT",
    database="YOUR_DB",
    user="YOUR_USER",
    password="YOUR_PASSWORD",
    port=5432
)

cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS login_activity (
    activity_id SERIAL PRIMARY KEY,
    username VARCHAR(50),
    login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    selected_model VARCHAR(50)
);
""")

conn.commit()
cur.close()
conn.close()

print("✅ Tables created successfully")
