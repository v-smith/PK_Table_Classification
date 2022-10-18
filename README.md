# PK Table Multilabel Classification Model

1. Model classes can be found: 
```
data_loaders/models.py
```

2. Dataloaders can be found:
```
data_loaders/table_data_loaders.py
data_loaders/bow_data_loaders.py
data_loaders/kfold_CV_data_loaders.py
data_loaders/bootstrap_data_loaders.py
```

3. To initially check model is working (check overfit to small sample): 
```
scripts/tableclass_debugFFNN.py
scripts/tableclass_debugBOW.py
scripts/tableclass_debugCNN.py
use with corresponding model config
```

4. To run model training and validation:
```
scripts/tableclass_trainFFNN.py
scripts/tableclass_trainBOW.py
scripts/tableclass_trainCNN.py
```

5. To run model testing:
```
scripts/tableclass_evaluateFFNN.py
scripts/tableclass_evaluateBOW.py
scripts/tableclass_evaluateCNN.py
```

6. To run k-fold cross validation:
```
scripts/kfold_CV.py
```

7. To run Bootstrap:
```
scripts/FFNN_bootstrap.py
```

8. To run TensorBoard:
```
cd data
tensorboard --logdir=runs --bind_all
```

9. To plot:
```
#output of kfold cross validation: 
scripts/plot_kfold.py
#loss and f1: 
scripts/plot_loss_f1.py
#To plot sensitivity analysis:
scripts/plot_sens_analysis.py 
```

12. To review labels vs model predictions (using Prodigy Software) and make corrections: 
```
prodigy review final-test-qual-assessment labels,predictions -v choice

#get reviewed database out as JSONL file 
prodigy db-out test-corrections > ./data/train-test-val/test-corrected.jsonl
```
