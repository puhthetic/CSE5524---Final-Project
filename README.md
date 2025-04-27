# CSE5524---Final-Project

**Prequisites**
You must have at least Python 3.10+. This project should be run on OSC ascend version 3.12.

You must install these dependencies:
```bash
pip install git+https://github.com/WildlifeDatasets/wildlife-datasets@develop
pip install git+https://github.com/WildlifeDatasets/wildlife-tools
pip install kagglehub
pip install timm
pip install torch
pip install numpy
pip install pandas
```

Change line 106 to epochs = 1 for 6 mins of training time. The longer the model trains for, the better the results.

submission.csv should be saved after the code executes and you can view the predictions for the query images.

To run the code, run:
`python cse_5524_enhanced.py`


Leaderboard Position (labeled Tyler Li):
![image](https://github.com/user-attachments/assets/61a6307f-e0a6-4d9c-876a-1a2bbc22b4ce)
