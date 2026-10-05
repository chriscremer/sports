"""One-time public SavedModel export; run under Python 3.12 + tf-keras 2.16.

Download the "New Weights" folder linked by AndrewDettor/TrackNet-Pickleball
into DATA/models/tracknet-pickleball first. Runtime inference needs no TensorFlow.
"""
import argparse
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--models',type=Path,
        default=Path('/Users/chriscremer/Downloads/pickleball_video_analysis/models'))
    args=p.parse_args()
    import tf_keras
    model=tf_keras.models.load_model(str(args.models/'tracknet-pickleball'),compile=False)
    (args.models/'tracknet-config.json').write_text(model.to_json())
    np.savez(args.models/'tracknet-weights.npz',
             **{f'{layer.name}__{i}':w for layer in model.layers for i,w in enumerate(layer.get_weights())})
    print('Exported unchanged Keras weights and graph for PyTorch inference.')

if __name__=='__main__':main()
