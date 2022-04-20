#!/usr/bin/env python3

import json
import glob
import os
import sqlite3

def _create_db():
    db_file = 'xcars.sqlite'
    os.remove(db_file)
    new_db = not os.path.exists(db_file)
    con = sqlite3.connect(db_file)
    cur = con.cursor()
    if new_db:
        print('creating tables')
        cur.execute(
        '''CREATE TABLE brands (
                id text PRIMARY KEY,
                name text NOT NULL UNIQUE
            )''')

        cur.execute(
        '''CREATE TABLE series (
                id integer PRIMARY KEY,
                brand_id text,
                name text NOT NULL,
                UNIQUE (brand_id, name),
                FOREIGN KEY (brand_id) REFERENCES brands (id)
            )''')

        cur.execute(
        '''CREATE TABLE models (
                id integer PRIMARY KEY,
                brand_id text,
                series_id integer,
                name text NOT NULL,
                filename text NOT NULL,
                UNIQUE (brand_id, series_id, name),
                FOREIGN KEY (brand_id) REFERENCES brands (id),
                FOREIGN KEY (series_id) REFERENCES series (id)
            )''')

        con.commit()
    return con, cur

con, cur = _create_db()

os.chdir('xcars_pages')

for cf in glob.glob('c_*.json'):
    with open(cf, 'r') as f:
        config_keys = set(k.strip() for k in json.load(f).keys())

b_sql = 'INSERT INTO brands (id, name) VALUES(?, ?)'
s_sql = 'INSERT INTO series (id, brand_id, name) VALUES(?, ?, ?)'
m_sql = 'INSERT INTO models (id, brand_id, series_id, name, filename) VALUES(?, ?, ?, ?, ?)'
with open('brands.json', 'r') as bf:
    brands_data = json.load(bf)
    for b in brands_data:
        b_name = b['name'].strip()
        b_id = b['id']
        cur.execute(b_sql, (b_id, b_name))

        s_filename = f's_{b_id}.json'
        if not os.path.exists(s_filename):
            print(f'no series under {b_name}')
            continue
        with open(s_filename, 'r') as sf:
            series_data = json.load(sf)
            for s in series_data:
                s_name = s['name'].strip()
                s_id = int(s['id'])
                cur.execute(s_sql, (s_id, b_id, s_name))

                m_filename = f'm_{s_id}.json'
                if not os.path.exists(m_filename):
                    print(f'no models under {s_name}')
                    continue
                with open(m_filename, 'r') as mf:
                    models_data = json.load(mf)
                    for m in models_data:
                        m_id = int(m)

                        c_filename = f'c_{m_id}.json'
                        assert os.path.exists(c_filename), f'{c_filename} doesn\'t exist'
                        with open(c_filename, 'r') as cf:
                            config_data = json.load(cf)
                            try:
                                filename = f'config_{b_id}_{s_id}_{m_id}.csv'
                                del config_data['zhuge']
                                m_name = config_data['name']
                                with open(filename, 'w') as f:
                                    f.write(f'厂商-车系-车型,{b_name}-{s_name}-{m_name}\n')
                                    del config_data['name']
                                    for k in sorted(config_data.keys()):
                                        f.write(f'{k},{config_data[k]}\n')
                                cur.execute(m_sql, (m_id, b_id, s_id, m_name, filename))
                                con.commit()
                            except sqlite3.IntegrityError as e:
                                if 'UNIQUE constraint failed' in str(e):
                                    print(f'WARNING: skipping {m_name}, there is an existing model with same brand and series')
                                else:
                                    raise e

con.commit()
cur.close()
con.close()
