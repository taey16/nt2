# nt2
- Modification of [neuraltalk2](https://github.com/karpathy/neuraltalk2)

# Features
- In our experiments, we got reached at CIDEr 0.8 with our BN-inception7 net(vgg16 baseline: 0.65~0.68). 
![alt tag](https://github.com/taey16/nt2/blob/master/logs/nt2_vgg_inception_loss.png)
![alt tag](https://github.com/taey16/nt2/blob/master/logs/nt2_vgg_inception_CIDEr.png)
We trained LSTM at first, and then finetuned CNN as well(we got about 0.02 points improvement but missing above curve). 

- Our best model reached at ~0.9 CIDEr
![alt tag](https://github.com/taey16/nt2/blob/master/logs/nt2_ResCeption_embedding2048_lstm3_loss.png)
![alt tag](https://github.com/taey16/nt2/blob/master/logs/nt2_ResCeption_embedding2048_lstm3_CIDEr.png)

# Acknowledgements
- Karpathy's great works [neuraltalk2](https://github.com/karpathy/neuraltalk2),[neuraltalk](https://github.com/karpathy/neuraltalk)

# Misc.
- our BN-inception7-Residual net: [inception7_residual.lua](https://github.com/taey16/image-encoder/blob/master/models/inception7_residual.lua), [inception_modules.lua](https://github.com/taey16/image-encoder/blob/master/models/inception_module.lua)

We welcome to any questions, discussion, issues. 
(taey1600@gmail.com)
