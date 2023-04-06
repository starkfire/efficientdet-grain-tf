# efficientdet-grain-tf

## Usage

### Requirements

* protobuf-compiler (protoc)
* CMake

### Setup

```sh
git clone https://github.com/starkfire/efficientdet-grain-tf

cd efficientdet-grain-tf/

# create a virtual environment within the project directory
virtualenv .venv
source .venv/bin/activate

# set up tensorflow/models
git clone --depth=1 https://github.com/tensorflow/models
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .

# go back to the project's root directory
cd ../../

# setup required directories
./setup_dirs.sh
```

### Prepare Dataset

Copy your test data and train data in `images/test` and `images/train` respectively, together with your annotations (`*.xml`) in PASCAL VOC format.

Run the `xml2csv.py` to aggregate your annotations in CSV files. This will create `images/train_labels.csv` and `images/test_labels.csv`.

```sh
python xml2csv.py
```

You also want to indicate labels in your dataset by modifying `annotations/label_map.pbtxt`.

```txt
item {
  id: 1,
  name: 'healthy'
}
item {
  id: 2,
  name: 'broken'
}
item {
  id: 3,
  name: 'damaged'
}
```

### TFRecords

Use `build_tfrecords.py` to set up TFRecords for your training and test data. This will generate `train.record` and `test.record` on the project's root directory.

Modify `build_tfrecords.py` so that `class_text_to_int()` contains your target labels:

```py
def class_text_to_int(row_label):
    if row_label == 'healthy':
        return 1
    if row_label == 'broken':
        return 2
    if row_label == 'damaged':
        return 3
    else:
        return None
```

Fetch the pre-trained EfficientDet weights (i.e. `efficientdet_d0_coco17_tpu-32`) and unpack its contents inside the `./pretrained` directory.

```sh
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
tar -xvf efficientdet_d0_coco17_tpu-32.tar.gz -C ./pretrained
```

## Troubleshooting
* protoc
```sh
apt install -y protobuf-compiler
```
* CMake
```sh
apt remove cmake -y
pip install cmake --upgrade
```