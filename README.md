Classify Tables from Pharmacokinetic Literature on PubMed using a MLP and CNN. 

To run tensor board: 
```
cd data
tensorboard --logdir=runs --bind_all
```

To review labels vs predictions and make corrections: 
```
prodigy review test-corrections labels,predictions -v choice
prodigy db-out test-corrections > ./data/train-test-val/test-corrected.jsonl

prodigy review val-corrections labels,predictions -v choice
prodigy db-out val-corrections > ./data/train-test-val/val-corrected.jsonl

prodigy review train-corrections labels,predictions -v choice
prodigy db-out train-corrections > ./data/train-test-val/train-corrected.jsonl
```
