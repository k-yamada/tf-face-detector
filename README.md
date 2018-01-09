# TF Face Detector

Face Detector using [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/object_detection)

![](https://user-images.githubusercontent.com/80381/29495837-2c0b05de-8602-11e7-8d38-c792e72e51d5.jpg)

Web DEMO: https://tf-face-detector.herokuapp.com/


## Prerequisite

- Python 2 (Cloud MLはpython3をサポートしていないため)
  - TensorFlow >= 1.2
  - Pillow >= 4.2.1 (for visualizing results)
  - cv2 >= 3.3 (for generating dataset)

## Setup

```
virtualenv --system-site-packages ~/tf-face-detector
source ~/tf-face-detector/bin/activate
git submodule update --init
pip install -r requirements.txt
```

## FDDB dataset

http://vis-www.cs.umass.edu/fddb/

To download data and generate tfrecord dataset (needed `cv2`):

```
python data/fddb.py
```


## Training

```sh
perl -pe "s|PATH_TO_BE_CONFIGURED|${PWD}/data|g" ./ssd_inception_v2_fddb.config.base > ssd_inception_v2_fddb.config
(cd models && protoc object_detection/protos/*.proto --python_out=.)
export PYTHONPATH=$PYTHONPATH:`pwd`/models:`pwd`/models/slim
python models/object_detection/train.py \
    --train_dir=./train \
    --pipeline_config_path=./ssd_inception_v2_fddb.config
```

or

```sh
./run_local.sh
```

## Training on Google Cloud Platform

### 1. Cloud Platform Console プロジェクトを作成とセットアップをする

参考: https://cloud.google.com/ml-engine/docs/command-line?hl=ja

上記を参考に、プロジェクトを作成して必要なセットアップをして下さい。
プロジェクト名は「tf-face-detector」にします。

### 2. モデル トレーニングやバッチ予測の実行中にデータを読み書きするための Google Cloud Storage バケットを作成する

参考: [Cloud Storage バケットを設定する](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction?hl=ja#set_up_your_cloud_storage_bucket)


```
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-mlengine2
REGION=asia-east1
# バケットを作成
gsutil mb -l $REGION gs://$BUCKET_NAME
```

### 3. データファイルを Cloud Storage バケットにアップロードします

1. gsutil を使用して data ディレクトリを Cloud Storage バケットにコピーします。

```
gsutil cp -r data gs://$BUCKET_NAME/data
```

### 4. configファイルを Cloud Storage バケットにアップロードします

```sh
perl -pe "s|PATH_TO_BE_CONFIGURED|gs://$BUCKET_NAME/data|g" ./ssd_inception_v2_fddb.config.base > ssd_inception_v2_fddb.config
gsutil cp -r ssd_inception_v2_fddb.config gs://$BUCKET_NAME/ssd_inception_v2_fddb.config
```

### 5. パッケージング

参考: [Packaging](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_cloud.md#packaging)

tensorflow/modelsをcloneします。

```
cd ~/git
git clone git@github.com:tensorflow/models.git
cd models/research/
```

`matplotlib`をrequireするように、`setup.py`を書き換えます。

参考: https://stackoverflow.com/a/47245648

```
# setup.py:

#REQUIRED_PACKAGES = ['Pillow>=1.0']
REQUIRED_PACKAGES = ['Pillow>=1.0','matplotlib']
```

パッケージングします

```
protoc object_detection/protos/*.proto --python_out=.
python setup.py sdist
(cd slim && python setup.py sdist)
```

### 6. クラウド内で単一インスタンスのトレーナーを実行する

#### 参考
- [クラウド内で単一インスタンスのトレーナーを実行する](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction?hl=ja#cloud-train-single)
- [Running an Evaluation Job on Cloud](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_cloud.md#running-an-evaluation-job-on-cloud)

```
JOB_NAME=job_`date +%Y%m%dT%I%M%S`
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
gcloud ml-engine jobs submit training $JOB_NAME \
--job-dir $OUTPUT_PATH \
--packages $HOME/git/models/research/dist/object_detection-0.1.tar.gz,$HOME/git/models/research/slim/dist/slim-0.1.tar.gz \
--module-name object_detection.eval \
--region $REGION \
--scale-tier BASIC_GPU \
-- \
--checkpoint_dir=$OUTPUT_PATH/train \
--eval_dir=$OUTPUT_PATH/eval \
--pipeline_config_path=gs://${BUCKET_NAME}/ssd_inception_v2_fddb.config
```

### 7. Running Tensorboard
You can run Tensorboard locally on your own machine to view progress of your training and eval jobs on Google Cloud ML. Run the following command to start Tensorboard:

```
tensorboard --logdir=gs://${TRAIN_DIR}
```

Note it may Tensorboard a few minutes to populate with results.
