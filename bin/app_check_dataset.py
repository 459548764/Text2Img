import numpy as np
import pandas as pd
import json, argparse
import text2img
from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import BadRequest, NotFound
import string
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


app = Flask(__name__)


def main(port,
         data_name,
         path_to_tfrecord,
         path_to_meta):

    model = text2img.SampleModel(
        path_to_tfrecord=path_to_tfrecord,
        path_to_meta=path_to_meta
    )

    @app.route('/')
    def main_endpoint():
        image, captions = model.get_data()

        plt.figure()
        plt.imshow(image, vmin=0, vmax=255)
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        plt.xticks(color="None")
        plt.yticks(color="None")

        # Rendering Plot in Html
        figfile = BytesIO()
        plt.savefig(figfile, format='png', bbox_inches="tight")
        figfile.seek(0)
        image = base64.b64encode(figfile.getvalue()).decode('ascii')

        return render_template('confirm_check_dataset.html',
                               image=image,
                               dataset=data_name,
                               captions=captions)

    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('--data', help='Dataset.', required=True, type=str, **share_param)
    parser.add_argument('--port', help='Port.', default=5000, type=int, **share_param)
    parser.add_argument('--valid', help='Validation data.', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_options(
        argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter))

    main(port=args.port,
         data_name=args.data,
         path_to_meta='./datasets/tfrecords/%s_meta.json' % args.data,
         path_to_tfrecord='./datasets/tfrecords/%s_%s.tfrecord' % (args.data, 'valid' if args.valid else 'train')
         )
