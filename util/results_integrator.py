import os
dataset = 'citeulike-a'

for i in sorted([x[2] for x in os.walk('citeulike-a/dbisma01')][0]):

for i  in sorted([x[1] for x in os.walk(dataset)][0]):