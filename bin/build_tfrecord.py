""" script to build tfrecord """

import text2img
import argparse


DATASET_PATH = dict(
    coco_2014=dict(
        valid=dict(
            image='./datasets/coco/images/val2014',
            annotation='./datasets/coco/annotations/annotations/captions_val2014.json'
        ),
        train=dict(
            image='./datasets/coco/images/train2014',
            annotation='./datasets/coco/annotations/annotations/captions_train2014.json'
        )
    ),
    coco_2017=dict(
        valid=dict(
            image='./datasets/coco/images/val2017',
            annotation='./datasets/coco/annotations/annotations/captions_val2017.json'
        ),
        train=dict(
            image='./datasets/coco/images/train2017',
            annotation='./datasets/coco/annotations/annotations/captions_train2017.json'
        )))


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('--data', help='Dataset.', required=True, type=str, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_options(
        argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter))

    target_data = DATASET_PATH[args.data]

    for data_type in ['train', 'valid']:  # build train data first to construct lookups
        text2img.create_tfrecord(
            path_to_image=target_data[data_type]['image'],
            path_to_annotation=target_data[data_type]['annotation'],
            path_to_save='./datasets/tfrecords/%s_%s' % (args.data, data_type),
            path_to_meta_dict='./datasets/tfrecords/%s_meta.json' % args.data)
