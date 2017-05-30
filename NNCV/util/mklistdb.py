import os
import sys


def makefile(filename):
    r = os.listdir()
    with open(filename, 'w', encoding='utf8') as f:
        for item in r:
            if item.endswith(".bin"):
                f.write('{}\n'.format(item))
 
 
if __name__ == '__main__':
    if len(sys.argv) == 2:
        makefile(sys.argv[1])
    else:
        makefile('ldatabatch.txt')
