import sqlite3

conn = sqlite3.connect("aoi.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS defects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    status TEXT NOT NULL,
    defect_type TEXT,
    station TEXT,
    shift TEXT,
    batch TEXT,
    image_path TEXT,
    scratch_mask_path TEXT,
    cap_mask_path TEXT,
    weld_json TEXT,
    model_version TEXT,
    operator TEXT
);
""")

conn.commit()
conn.close()

print("defects 表已建立！")
