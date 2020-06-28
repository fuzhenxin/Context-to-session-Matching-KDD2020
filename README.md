# Context-to-Session Matching
This repo contains the code and data for paper **Context-to-Session Matching: Utilizing Whole Session for Response Selection in Information-Seeking Dialogue Systems** in KDD 2020.

## Code
1. Preprocess: ```cd utils ; python compose_data.py```
2. The configure is located in main.py (detailed introductions are in main.py)
3. How to train:
```python main.py train```
4. How to test:
```python main.py test TestRandNegCand $chenkpoint_file```

## Data Structure
1. The data sets are located in data_ali directory.
    - cc.cc.train(.zip) represents the train set for CSM, please uncompress it first.
    - cc.cr.train(.zip) represents the train set for CRM, please uncompress it first.
    - cc.cc.{valid}/{test}/{test.human} represent the valid/TestRandNegCand test/TestRetrvCand test set for CSM and CRM
    - vectors.txt contains the pre-trained word embedding.
2. Format of each line in the files: query context|response context|response|label
3. Please note that: the labels in cc.cc.test.human are all 1 and the right human annotated labels of TestRetrvCand are in human/human.res.crowd


## ACK
The code is developed referring [DAM](https://github.com/baidu/Dialogue/tree/master/DAM).
