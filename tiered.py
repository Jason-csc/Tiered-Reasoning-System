import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import progressbar
import gc


def saveModel(model,epoch,weight,lr,acc1,acc2,DRIVE_PATH):
  path = f'{DRIVE_PATH}/model/tieredModel_w0{weight[0]}_w1{weight[1]}_lr{lr}_acc1{acc1}_acc2{acc2}_epoch{epoch}.torch'
  print("\nsaving model to",path)
  torch.save(model.state_dict(),path)


def loadModel(ROBERTA_PATH,PATH,device):
    tieredModel = TieredModel(ROBERTA_PATH=ROBERTA_PATH)
    tieredModel.to(device)
    tieredModel.load_state_dict(torch.load(PATH))
    return tieredModel