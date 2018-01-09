import cv2
import math
import os
import tarfile

import numpy as np
import tensorflow as tf

IMAGES_DOWNLOAD_URL = 'http://tamaraberg.com/faceDataset/originalPics.tar.gz'
ANNOTATIONS_DOWNLOAD_URL = 'http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz'

DIRECTORY = os.path.join(os.path.dirname(__file__), 'fddb')

# CASCADES_DIR = os.path.normpath(os.path.join(cv2.__file__, '..', '..', '..', '..', 'share', 'OpenCV', 'haarcascades'))
CASCADES_DIR = "/usr/local/opt/opencv/share/OpenCV/haarcascades"
FACE_CASCADE = cv2.CascadeClassifier(os.path.join(CASCADES_DIR, 'haarcascade_frontalface_default.xml'))
EYES_CASCADE = cv2.CascadeClassifier(os.path.join(CASCADES_DIR, 'haarcascade_eye.xml'))

EXCLUDES = set([
    '2002/07/19/big/img_445',
    '2002/07/21/big/img_76',
    '2002/07/22/big/img_152',
    '2002/07/26/big/img_513',
    '2002/07/29/big/img_136',
    '2002/07/31/big/img_898',
    '2002/08/05/big/img_3591',
    '2002/08/07/big/img_1576',
    '2002/08/16/big/img_81',
    '2002/08/16/big/img_1055',
    '2002/08/18/big/img_293',
    '2002/08/21/big/img_516',
    '2003/01/01/big/img_547',
    '2003/01/14/big/img_951',
    '2003/01/15/big/img_640',
    '2003/01/15/big/img_17',
])

flags = tf.app.flags
flags.DEFINE_string('output_dir', os.path.dirname(__file__), 'Path to directory to output TFRecords.')
FLAGS = flags.FLAGS


def download_and_extract():
    download = tf.contrib.learn.datasets.base.maybe_download
    for target in [IMAGES_DOWNLOAD_URL, ANNOTATIONS_DOWNLOAD_URL]:
        filename = os.path.basename(target)
        filepath = download(filename, DIRECTORY, target)
        with tarfile.open(filepath) as tar:
            for file in tar:
                path = os.path.join(DIRECTORY, file.name)
                if not os.path.exists(path):
                    print('extract {}'.format(path))
                    tar.extract(file, path=DIRECTORY)


def detect_faces(img, lines):
    results = []
    for line in lines:
        e = line.split(' ')
        size = max(float(e[0]), float(e[1])) * 1.1
        # skip if face is too small
        if size < 60.0:
            break
        # crop to detect frontalface
        center = (int(float(e[3]) + .5), int(float(e[4]) + .5))
        angle = float(e[2]) / math.pi * 180.0
        if angle < 0:
            angle += 180.0
        M = cv2.getRotationMatrix2D(center, angle - 90.0, 1)
        M[0, 2] -= float(e[3]) - size
        M[1, 2] -= float(e[4]) - size
        target = cv2.warpAffine(img, M, (int(size * 2 + .5), int(size * 2 + .5)))

        # detect face and eyes
        faces = FACE_CASCADE.detectMultiScale(target)
        if len(faces) != 1:
            print('{} faces found...'.format(len(faces)))
            break
        face = faces[0]
        face_img = target[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
        eyes = []
        for eye in EYES_CASCADE.detectMultiScale(face_img):
            # reject false detection
            if eye[1] > face_img.shape[0] / 2:
                break
            eyes.append(eye)
        if len(eyes) != 2:
            print('{} eyes found...'.format(len(eyes)))
            break
        # reject invalid scale eyes
        if not (2. / 3. < eyes[0][2] / eyes[1][2] < 3. / 2. and 2. / 3. < eyes[0][3] / eyes[1][3] < 3. / 2.):
            break

        # calculate coordinates of the original image
        center_points = [[
            face[0] + face[2] / 2.0,
            face[1] + face[3] / 2.0,
        ], [
            face[0] + eyes[0][0] + eyes[0][2] / 2.0,
            face[1] + eyes[0][1] + eyes[0][3] / 2.0,
        ], [
            face[0] + eyes[1][0] + eyes[1][2] / 2.0,
            face[1] + eyes[1][1] + eyes[1][3] / 2.0,
        ]]
        p = np.hstack([np.array(center_points), np.ones((3, 1))])
        p = cv2.invertAffineTransform(M).dot(p.T).T
        results.append([{
            'class': 'face',
            'xmin': p[0][0] - face[2] / 2.0,
            'xmax': p[0][0] + face[2] / 2.0,
            'ymin': p[0][1] - face[3] / 2.0,
            'ymax': p[0][1] + face[3] / 2.0,
        }, {
            'class': 'eye',
            'xmin': p[1][0] - eyes[0][2] / 2.0,
            'xmax': p[1][0] + eyes[0][2] / 2.0,
            'ymin': p[1][1] - eyes[0][3] / 2.0,
            'ymax': p[1][1] + eyes[0][3] / 2.0,
        }, {
            'class': 'eye',
            'xmin': p[2][0] - eyes[1][2] / 2.0,
            'xmax': p[2][0] + eyes[1][2] / 2.0,
            'ymin': p[2][1] - eyes[1][3] / 2.0,
            'ymax': p[2][1] + eyes[1][3] / 2.0,
        }])
    return results


def write_record(writer, img, filepath, data):
    h, w, _ = img.shape
    xmin, xmax, ymin, ymax = [], [], [], []
    class_text, class_label = [], []
    label_map_dict = {
        'face': 1,
        'eye': 2,
    }
    for face in data:
        for bbox in face:
            label = label_map_dict[bbox['class']]
            xmin.append(bbox['xmin'] / w)
            xmax.append(bbox['xmax'] / w)
            ymin.append(bbox['ymin'] / h)
            ymax.append(bbox['ymax'] / h)
            class_text.append(bbox['class'].encode('utf-8'))
            class_label.append(label)
    with open(filepath, 'rb') as f:
        encoded = f.read()
    feature = {
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filepath.encode('utf-8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filepath.encode('utf-8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf-8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=class_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=class_label)),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


def main(argv=None):
    download_and_extract()
    writers = {
        'train': tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_dir, 'fddb_train.record')),
        'val': tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_dir, 'fddb_val.record')),
    }
    folds_dir = os.path.join(DIRECTORY, 'FDDB-folds')
    for filename in os.listdir(folds_dir):
        if not filename.endswith('-ellipseList.txt'):
            continue
        writer = writers['val'] if 'fold-10' in filename else writers['train']
        with open(os.path.join(folds_dir, filename)) as f:
            for line in f:
                img_file = line.strip()
                print(img_file)
                if len(img_file) == 0:
                    break
                num = int(f.readline())
                lines = []
                for _ in range(num):
                    lines.append(f.readline().strip())
                if img_file in EXCLUDES:
                    continue
                # load image, detect faces
                filepath = os.path.join(DIRECTORY, '{}.jpg'.format(img_file))
                img = cv2.imread(filepath)
                detected = detect_faces(img, lines)
                # skip if all faces waren't detected
                if len(detected) != len(lines):
                    continue
                write_record(writer, img, filepath, detected)
                # for results in detected:
                #     for obj in results:
                #         cv2.rectangle(
                #             img,
                #             tuple([int(obj['xmin'] + .5), int(obj['ymin'] + .5)]),
                #             tuple([int(obj['xmax'] + .5), int(obj['ymax'] + .5)]),
                #             [0, 255, 0] if obj['class'] == 'face' else [255, 255, 0]
                #         )
                # cv2.imshow(img_file, img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
    for writer in writers.values():
        writer.close()


if __name__ == '__main__':
    tf.app.run()
