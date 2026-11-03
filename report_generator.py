import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect("aoi.db")
df = pd.read_sql_query("SELECT * FROM defects", conn)

# 缺陷 Pareto
pareto = df[df["status"]=="NG"]["defect_type"].value_counts()
pareto.plot(kind="bar")
plt.title("Defect Pareto")
plt.savefig("pareto.png")

# 班別不良率
shift_rate = df.groupby("shift")["status"].apply(lambda x: (x=="NG").mean())
shift_rate.plot(kind="bar")
plt.title("Shift NG Rate")
plt.savefig("shift_rate.png")
