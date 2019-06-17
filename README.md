# Multi-Representation Fusion Network (MRFN)

- This is an implementation of [Multi-Representation Fusion Network for Multi-turn Response Selection in Retrieval-based Chatbots].


## Requirements
* Python 2.7
* NumPy
* Tensorflow 1.4.0


## Usage
To download and preprocess the data, run

```bash
# download ubuntu corpus and pre-trained word/char embeddings and dictionaries
sh download.sh
# preprocess the data
python data_utils_record.py
```

All hyper parameters are stored in config.py. To train and valid, run

```bash
python main.py --log_root=logs_ubuntu --batch_size=100
```

To evaluate the model, run
```bash
python evaluate.py --log_root=logs_ubuntu --batch_size=100
```