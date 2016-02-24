# nt2
- Modification of [neuraltalk2](https://github.com/karpathy/neuraltalk2)

# Features
- In our experiments, we got reached at CIDEr 0.8 with our BN-inception7 net(vgg16 baseline: 0.65~0.68). 
![alt tag](https://github.com/taey16/nt2/blob/master/logs/nt2_vgg_inception_loss.png)
![alt tag](https://github.com/taey16/nt2/blob/master/logs/nt2_vgg_inception_CIDEr.png)
We trained LSTM at first, and then finetuned CNN as well(we got about 0.02 points improvement but missing above curve). 
- Our best model reached at 0.923 CIDEr

![alt tag](https://github.com/taey16/nt2/blob/master/logs/nt2_loss_inception-v3_embedding2048_hidden384_layer3.png)
![alt tag](https://github.com/taey16/nt2/blob/master/logs/nt2_CIDEr_inception-v3_embedding2048_hidden384_layer3.png)

1. We used inception-v3-residual net not vgg. 
2. [We removed random embedding(projection)](https://github.com/taey16/nt2/blob/master/misc/net_utils.lua#L28-L32)
3. We used 3 layered deep LSTM

# Acknowledgements
- Karpathy's great works [neuraltalk2](https://github.com/karpathy/neuraltalk2),[neuraltalk](https://github.com/karpathy/neuraltalk)

# Misc.
- our Inception-Residual net: [resception.lua](https://github.com/taey16/image-encoder/blob/master/models/resception.lua), [inception_modules.lua](https://github.com/taey16/image-encoder/blob/master/models/inception_module.lua), [inception-v3.lua](https://github.com/taey16/image-encoder/blob/master/models/inception_v3.lua)

We welcome to any questions, discussion, issues. 
(taey1600@gmail.com)
