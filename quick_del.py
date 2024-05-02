import os

file = open('del.txt', 'r')
lines = file.readlines()

for str_dir in lines:
    str_dir = "".join(map(lambda x: str.strip(x), str_dir))
    os.remove(str_dir)