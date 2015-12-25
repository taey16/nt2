
require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
path = require 'pl.path'
require 'misc.DataLoader'
utils = require 'misc.utils'
net_utils = require 'misc.net_utils'

--opt = paths.dofile('opt/opt_coco_vgg.lua')
opt = paths.dofile('opt/opt_coco_inception7.lua')

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.manualSeed(opt.seed)
--cutorch.setDevice(opt.gpuid + 1)

loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}
protos, thin_lm, thin_cnn = paths.dofile('model.lua')

paths.dofile('train.lua')
paths.dofile('test.lua')

local iter = 0
local loss0
optim_state = {}
cnn_optim_state = {}
loss_history = {}
val_lang_stats_history = {}
val_loss_history = {}
best_score = nil
local tm = torch.Timer()

if opt.test_initialization then 
  local losses_val, predictions_val, lang_stats = test(iter) 
  print(('initial validation loss: %.6f'):format(losses_val))
  iter = iter+1 
else iter = iter+1 end

while true do  
  local elapsed_trn = tm:time().real
  local losses_trn, finetune_cnn = train(iter)
  io.flush(print(string.format(
    'iter %d trn loss: %f, finetune: %s, %.4f', 
      iter, losses_trn.total_loss, 
      tostring(finetune_cnn), tm:time().real - elapsed_trn
  )))
  if iter % opt.losses_log_every == 0 then 
    loss_history[iter] = losses_trn.total_loss 
  end

  if iter % opt.save_checkpoint_every == 0 or 
     iter == opt.max_iters then
    local elapsed = tm:time().real
    local losses_val, predictions_val, lang_stats = test(iter)
    print(lang_stats)
    print(string.format(
      'validation loss: %.6f, %.2f', 
      losses_val, tm:time().real - elapsed))
    val_loss_history[iter] = losses_val
    if lang_stats then
      val_lang_stats_history[iter] = lang_stats
    end
    conditional_save(iter, losses_val, predictions_val, lang_stats)
  end
  if check_termination(iter, loss0, losses_trn) then
    break
  end
  iter = iter + 1
end
print('===> End of training')

