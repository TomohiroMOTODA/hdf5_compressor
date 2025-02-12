'''
* ``Author``: Tomohiro MOTODA
* ``Organization``: AIST

* Usage
  ```console
  $ python -m hdf5_compressor.create_text_embeddings
  ```
  It outputs text_embeddings.npy at ./static (overlapped), loading ./static/feature.csv.
  When you add something new int the database, describe the task meta infomation at feature.csv (please follow feature.cvs)
'''
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

def main ():    

    df = pd.read_csv('./static/feature.csv')
    tasks = df['Task Name'].to_list()

    texts_dict = dict()
    for task_name in tqdm(tasks, total=len(tasks)):

        try:
            # --- load metadata --- #
            info = df[df['Task Name']==task_name]

            # --- text embeddings --- #
            os.environ["TOKENIZERS_PARALLELISM"] = "false" # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
            model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = info["Text"].to_numpy()
            embeddings = model.encode(texts[0])
            texts_dict[task_name] = embeddings
        except TypeError as e:
            print('catch TypeError:', e)

    conf_path = os.path.join("./static", f'text_embeddings.npy')
    np.save (conf_path, np.array(texts_dict))

# sample
def load ():
    conf_path = os.path.join("./static", f'text_embeddings.npy')
    texts = np.load(conf_path, allow_pickle=True)
    texts = texts.item()
    print (texts['xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'])

# sample
def sample_code():
    # Load a pretrained Sentence Transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # The sentences to encode
    sentences = [
        "bluebox_delivery",
        "gear_task/01_washer_len700",
        "gear_task/02_gear_len500",
        "insert_usb_cable",
        "insert_usb_cable_fix_parts",
        "gear_task_003_01",
        "gear_task_003_02",
        "cable_routing_8pin_cross",
        "cable_routing_cross_narrow",
        "color_tape_grab_and_insert",
        "goods_grab_and_insert",
        "silver_pouch_001",
        "silver_pouch_002",
        "silver_pouch_004",
        "clear_zipper_bag_001",
        "clear_zipper_bag_reel_001",
        "clear_zipper_bag_reel_002",
        "receive_goods_and_add_to_basket",
        "tape_placed_in_random_places_long",
    ]
    embeddings = model.encode(sentences)

if __name__=="__main__":
    main()