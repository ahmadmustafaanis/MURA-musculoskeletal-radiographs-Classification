# Murav1.1

## To Run

* Create a folder named input
* Create a folder named src
* Download all the data in the input, so the directory looks like
```
src/
|    models/
|    dataloaders.py
|    preprocessor.py
|    run.py
|    readme.md
________________
input/
|    murav1.1/
|    |    MURA-v1.1/
|    |    |    train/
|    |    |    valid/
|    |    |    train_image_paths.csv
|    |    |    train_labelled_studies.csv
|    |    |    valid_image_paths.csv
|    |    |    valid_labelled_studies.csv
________________
```

* Go to the src directory first and the run `run.py`
```
$ cd src
$ python run.py --data_path ../input/mura-v1.1/MURA-v1.1/
```

You can specify batch size, number of epochs, saved model name via folowing
```
$ python run.py --data_path ../input/mura-v1.1/MURA-v1.1/ --epochs 100 --model_path models/modelname.h5 --batchsize 16
```