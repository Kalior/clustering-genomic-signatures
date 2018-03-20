import matplotlib.pyplot as plt
import os

from vlmc import VLMC
from parse_trees_to_json import parse_trees
from get_signature_metadata import get_metadata_for


def save(tree_dir, out_dir):
  parse_trees(tree_dir)
  vlmcs = VLMC.from_json_dir(tree_dir)
  metadata = get_metadata_for([vlmc.name for vlmc in vlmcs])

  try:
    os.stat(out_dir)
  except:
    os.mkdir(out_dir)

  for vlmc in vlmcs:
    plt.figure(figsize=(30, 20), dpi=80)
    vlmc.draw(metadata)
    out_file = os.path.join(out_dir, vlmc.name + '.svg')
    plt.savefig(out_file, dpi='figure', format='svg')
    plt.close()

if __name__ == '__main__':
  tree_dir = '../trees_pst_better'
  image_dir = '../images'
  save(tree_dir, image_dir)
