# VAD
Repository contains voice activity detection algorithms and datasets

`links.txt` contains all links related to project: excel with frame ids, recordings and other
`vad_env.yml` is a image of the conda enviroment. User can install the whole enviroment by anaconda's prompt commend `conda env create -f vad_env.yml`

Dir `analyze_wav` contains `take_frame_by_number.py` file which can be called with parser. To see the usage use the command `python take_frame_by_number.py -h`

Link to dataset saved in `.json` format is stored in `json\database_link.txt`.

Dataset for each 20ms frame contains:
  > unique id <br>
  > raw samples <br>  
  > melspectrogram <br> 
  > mfcc
