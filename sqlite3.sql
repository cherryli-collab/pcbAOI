CREATE TABLE defects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME,
    status TEXT,
    defect_type TEXT,
    station TEXT,
    shift TEXT,
    batch TEXT,
    image_path TEXT
);
