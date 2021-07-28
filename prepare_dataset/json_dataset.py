from os.path import dirname, abspath, join
import sys

DIR_PATH = dirname(dirname(abspath(__file__)))
sys.path.append(DIR_PATH)

DATA_PATH = "wav\long\data_long.wav"
PATH = join(DIR_PATH,  DATA_PATH)

if __name__ == '__main__':
    
    