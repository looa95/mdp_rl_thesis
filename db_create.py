import sqlite3

conn = sqlite3.connect("radio_model.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS snr_lookup (
    resource INTEGER NOT NULL,
    tx_power_dbm INTEGER NOT NULL,
    dist_min REAL NOT NULL,
    dist_max REAL NOT NULL,
    snr_db REAL NOT NULL
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS params (
    key TEXT PRIMARY KEY,
    value REAL
);
""")

# Global threshold
cur.execute("INSERT OR REPLACE INTO params (key, value) VALUES ('snr_success_thresh_db', 5.0);")

# Populate toy values
resources = [0,1,2]
power_bins = [18,25,33]
dist_bins = [(0,30,20.0),(30,60,12.0),(60,120,6.0),(120,200,2.0),(200,10000,-3.0)]
power_gain = {18:0.0, 25:5.0, 33:10.0}

for r in resources:
    for p in power_bins:
        for (dmin,dmax,base_snr) in dist_bins:
            snr = base_snr + power_gain[p]
            cur.execute("INSERT INTO snr_lookup VALUES (?,?,?,?,?)", (r,p,dmin,dmax,snr))

conn.commit()
conn.close()
