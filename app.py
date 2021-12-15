from flask import (
  Flask, request, Response,
)

import torch
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision
import BigGAN_utils.utils as utils
import clip
import torch.nn.functional as F
from DiffAugment_pytorch import DiffAugment
import numpy as np
from fusedream_utils import FuseDreamBaseGenerator, get_G, save_image

app = Flask(__name__)

def generate(sentence, init_iters, opt_iters, num_basis, model, seed):
  utils.seed_rng(seed)
  print('Generating: ', sentence)
  if model == "biggan-256":
    G, config = get_G(256)
  elif model == "biggan-512":
    G, config = get_G(512)
  else:
    raise Exception("Model Not Supported")
  
  generator = FuseDreamBaseGenerator(G, config, 10)
  z_cllt, y_cllt = generator.generate_basis(sentence, init_iters=init_iters, num_basis=num_basis)

  z_cllt_save = torch.cat(z_cllt).cpu().numpy()
  y_cllt_save = torch.cat(y_cllt).cpu().numpy()
  img, z, y = generator.optimize_clip_score(z_cllt, y_cllt, sentence, latent_noise=False, augment=True, opt_iters=opt_iters, optimize_y=True)
  score = generator.measureAugCLIP(z, y, sentence, augment=True, num_smaples=20)
  print('AugCLIP score: ', score)
  import os
  if not os.path.exists('./samples'):
    os.mkdir('./samples')
  save_image(img, './samples/fusedream_%s_seed_%d_score_%.4f.png'%(sentence, seed, score))
  

@app.route('/fusedream', methods=['POST'])
def generate_fusedream():
  try:
    params = request.get_json()
    sentence = params['sentence']
    init_iters = params['init_iters']
    opt_iters = params['opt_iters']
    num_basis = params['num_basis']
    model = params['model']
    seed = params['seed']
  except Exception:
    return Response("Empty Field", status=400)
  
  generate(sentence, init_iters, opt_iters, num_basis, model, seed)


if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port="5000")