from scipy.io import wavfile
from os.path import dirname, abspath, join
import sys
import argparse

parser = argparse.ArgumentParser(description = "Cut out desired frame by id and save it to temp directory with the id number.")
parser.add_argument('-i', '--index', help='provide id of frames you want to save. use comma as delimiter. Indexing start with zero.', type=str)
parser.add_argument('-l', '--length', type=int, metavar='', help='length of frame in secs. Default value 0.2 s')
parser.add_argument('-m', '--my', type=str, metavar='', help='Cut your fragment of signal by providing first and last frame id. The third argument is the name of file. Args seperated by coma')
args = parser.parse_args()

d = dirname(dirname(abspath(__file__)))
sys.path.append(d)

LONG_WAV_PATH = "wav\long\Paciorek_Duda.wav"
TEMP_DIR = "temp"

# Frame size in sec
FRAME_SIZE = 0.2

if __name__ == '__main__':

    # check if another frame length
    if args.length is not None:
        FRAME_SIZE = args.length

    INPUT_PATH = join(d,LONG_WAV_PATH)
    OUTPUT_DIR = join(d, TEMP_DIR)

    SR, data = wavfile.read(INPUT_PATH)

    print("Do przetworzenia jest {} ramek".format(int(len(data) / (FRAME_SIZE * SR))))

    # SET number of samples in one frame
    SAMPLES_PER_FRAME = int(FRAME_SIZE * SR)
    
    if args.index is not None:
        frame_ids = [int(item) for item in args.index.split(',')]
        for id in frame_ids:
            frame = data[int(id*SAMPLES_PER_FRAME) : int((id + 1)*SAMPLES_PER_FRAME)]
            wavfile.write(join(OUTPUT_DIR, str(id)) + '.wav', SR, frame)

        print("Przetworzono {} ramek o długości {} s każda".format(len(frame_ids), FRAME_SIZE))

    if args.my is not None:
        
        arguments = [item for item in args.my.split(',')]
        name = str(arguments[-1])
        start = int(arguments[0])
        stop = int(arguments[1])
        frame = data[start : int(stop + 1)]

        frame = data[int(start * SAMPLES_PER_FRAME) : int((stop + 1) * SAMPLES_PER_FRAME)]
        
        wavfile.write(join(OUTPUT_DIR, name + '.wav'), SR, frame)
        print("Zapisano  {} ".format(join(OUTPUT_DIR, name + '.wav')))

    print('----------------------- END --------------------------------')
