"""Evalute the features for a submission using a pretrained model"""
import torch
import pandas as pd
import numpy as np

# load in data and make a pytorch tensor
df_preds = pd.read_csv('Sumbission_Features.csv')
X_preds = torch.from_numpy(df_preds.values)
#-----------------------------Load in Model----------------------------------
# with pytorch, model class must be defined to load in (dumb) 
# Use the nn package to define our model and loss function.

# NOTE always make sure this matches the model in pytorch_MLP.py
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 50, xDim, xDim, 1
model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.Dropout(p=0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        )
model.load_state_dict(torch.load('model.ckpt'))
# put model in eval mode
model.eval()

preds = model(X_pred)

df_sample_sub = pd.DataFrame()
# clipped_preds = np.clip(preds, 0.05, 0.95)
df_sample_sub.Pred = preds
df_sample_sub.shape


filename = 'makeBrack'
save_dir = './'
c=0
ext = '.csv'
if os.path.exists(save_dir+filename+ext):
    while os.path.exists(filename+ext):
        c+=1
    filename = filename+'_'+str(c)
    df_sample_sub.to_csv(save_dir+filename+ext, index=False)
else:
    df_sample_sub.to_csv(save_dir+filename+ext, index=False)


