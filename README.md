# FEDCSIS VAD
Repository contains voice activity detection algorithms and datasets

`links.txt` contains all links related to project: excel with frame ids, recordings and other

The script can be run by calling the `vad_experiment_driver.py` script. Outcomes of the script were used to prepare a FEDCSIS submission available at the URL below:<br/>
https://www.overleaf.com/2827146645cccvpbhqxgwb

To train the CNN make the call below:<br/>
*python vad_experiment_driver.py -es ann_training*

To inspect structure of the input data and evaluate accuracies on the training and the validation datasets make the call below:<br/>
*python vad_experiment_driver.py -es measure_train_val_acc*

To perform the evaluation bein the main part of the FEDCSIS submission, use the following call:<br/>
*python vad_experiment_driver.py -es model_evaluation*

# Publications
Results of the experiments were published during the FEDCSIS conference. The article is available [here](https://annals-csis.org/proceedings/2021/drp/146.html)
