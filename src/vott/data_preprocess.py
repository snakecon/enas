# coding=utf-8
# ==============================================================================
"""
Sample converter.
"""
import cPickle as pickle

import cv2
import numpy as np
import simplejson
from tqdm import tqdm


class SampleConverter(object):
    """sample converter."""

    def __init__(self):
        self.label2index = {
            'active': 0,
            'deactive': 1,
        }

    def convert(self, video_file_name, output_file_name):
        video_json_file_name = '%s.json' % video_file_name

        max_frame, video_meta = self.etract_meta(video_json_file_name)

        data = self.extact_image(video_file_name, max_frame, video_meta)

        self.store_data(data, output_file_name)

    def store_data(self, data, output_file_name):
        with open(output_file_name, 'w') as f:
            pickle.dump(data, f)

    def extact_image(self, video_file_name, max_frame, video_meta):
        video_capture = cv2.VideoCapture(video_file_name)

        fps = video_capture.get(cv2.CAP_PROP_FPS)
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        max_count = min(max_frame, count)

        image_data = []
        label_data = []

        frame_index = 0
        with tqdm(total=max_count, desc='video') as progress_bar:
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if ret is True:
                    frame_index += 1
                    if frame_index > max_count:
                        break
                    progress_bar.update(1)
                    if frame_index not in video_meta:
                        continue

                    label = video_meta[frame_index]

                    # Image for like HWC
                    image_data.append(np.array(frame))
                    label_data.append(self.label2index[label])
                    print(frame_index, label, frame.shape)
                else:
                    break

        # When everything is done, release the capture
        video_capture.release()

        return {'data': np.array(image_data), 'labels': label_data}

    def etract_meta(self, video_json_file_name):
        video_meta = {}
        vott_result = self.load_vott_result(video_json_file_name)
        for key, body in tqdm(vott_result['frames'].items(), desc='meta'):
            if len(body) == 0:
                continue
            tags = body[0]['tags']
            if len(tags) != 1:
                continue
            frame_tag = tags[0]
            frame_index = int(key)
            video_meta[frame_index] = frame_tag
        max_frame = max(video_meta.keys())
        return max_frame, video_meta

    def load_vott_result(self, video_json_file_name):
        with open(video_json_file_name) as json_file:
            json_content = json_file.read()
            return simplejson.loads(json_content)

    def read_data(self, sample_file_name):
        """Reads data. Always returns NHWC format.

        Returns:
          images: np tensor of size [N, H, W, C]
          labels: np tensor of size [N]
        """
        images, labels = [], []

        with open(sample_file_name, 'rb') as finp:
            data = pickle.load(finp)
            batch_images = data["data"].astype(np.float32) / 255.0
            batch_labels = np.array(data["labels"], dtype=np.int32)
            images.append(batch_images)
            labels.append(batch_labels)

        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)
        images = np.reshape(images, [-1, 360, 480, 3])

        return images, labels


if __name__ == '__main__':
    converter = SampleConverter()

    converter.convert('tpr/train.mp4', 'data/train_1')
    converter.convert('tpr/test.mp4', 'data/test_1')

    converter.read_data('data/train_1')
