from collections import namedtuple

Student = namedtuple('Student', 'Name, Roll, Class')

import csv

for s in map(Student._make, csv.reader(open('/Users/aniron/Documents/test.csv', 'rb'))):
    print s.Name, s.Roll, s.Class

for r in csv.reader(open('/Users/aniron/Documents/test.csv', 'rb')):
    print r