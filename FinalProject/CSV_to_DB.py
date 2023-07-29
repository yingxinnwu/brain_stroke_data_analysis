# Description: this file creates a database from brain_stroke.csv and creates a table called brain_stroke
# the table brain_stroke has the following columns: gender, age, avg_glucose_level, bmi, stroke

import sqlite3 as sl
import pandas as pd

# Create/Connect database
conn = sl.connect('brain_stroke.db')
curs = conn.cursor()

# Create our table if it doesn't already exist
# Manually specify table name, column names, and columns types
curs.execute('DROP TABLE IF EXISTS brain_stroke')
curs.execute('CREATE TABLE IF NOT EXISTS '
             'brain_stroke (`gender` text, `age` number, `avg_glucose_level` number, `bmi` number, `stroke` number)')
conn.commit()  # don't forget to commit changes before continuing

# Use pandas to read the csv to df
# Select just the columns you want using optional use columns param
df = pd.read_csv('brain_stroke.csv', usecols=['gender', 'age', 'avg_glucose_level', 'bmi', 'stroke'])
print('First 3 df results:')
print(df.head(3))

# Let pandas do the heavy lifting of converting a df to a db
# name=your existing empty db table name
# con=your db connection object
# just overwrite if the values already there and don't index any columns
df.to_sql(name='brain_stroke', con=conn, if_exists='replace', index=False)

print('\nFirst 3 db results:')
results = curs.execute('SELECT * FROM brain_stroke').fetchmany(3)
for result in results:
    print(result)

result = curs.execute('SELECT COUNT(*) FROM brain_stroke').fetchone()
# Note indexing into the always returned tuple w/ [0]
# even if it's a tuple of one
print('\nNumber of valid db rows:', result[0])
print('Number of valid df rows:', df.shape[0])

result = curs.execute('SELECT MAX(`avg_glucose_level`) FROM brain_stroke').fetchone()
print('Greatest average glucose level', result[0])

# number = 228.69
# v = (number,)
# result = curs.execute('SELECT `age` FROM brain_stroke WHERE `avg_glucose_level` = ?', v)
# for thing in result:
#     print(thing)
