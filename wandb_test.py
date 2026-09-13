import sqlite3
import json

db_path = "/data1/hemanth/mesh-splatting/wandb/run-20260824_202645-gtl4q8e8/run-gtl4q8e8.wandb"
conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", c.fetchall())

try:
    c.execute("SELECT * FROM wandb_history LIMIT 1;")
    names = [description[0] for description in c.description]
    print("wandb_history columns:", names)
    print("Sample row:", c.fetchone())
except Exception as e:
    print(e)
