
paths.dofile('lossFun.lua')
require 'misc.optim_updates'

-- flatten and prepare all model parameters to a single vector. 
-- Keep CNN params separate in case we want to try to get fancy with different optims on LM/CNN
params, grad_params = protos.lm:getParameters()
cnn_params, cnn_grad_params = protos.cnn:getParameters()
print('total number of parameters in LM: ', params:nElement())
print('total number of parameters in CNN: ', cnn_params:nElement())
assert(params:nElement() == grad_params:nElement())
assert(cnn_params:nElement() == cnn_grad_params:nElement())


local function lr_policy(iter)
  -- decay the learning rate for both LM and CNN
  local learning_rate = opt.learning_rate
  local cnn_learning_rate = opt.cnn_learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
    cnn_learning_rate = cnn_learning_rate * decay_factor
  end
  return learning_rate, cnn_learning_rate
end


local function update_grad(finetune_cnn, learning_rate, cnn_learning_rate)
  local cnn_learning_rate = cnn_learning_rate or nil
  -- perform a parameter update
  if opt.optim == 'rmsprop' then
    rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'adagrad' then
    adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'sgd' then
    sgd(params, grad_params, opt.learning_rate)
  elseif opt.optim == 'sgdm' then
    sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'sgdmom' then
    sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'adam' then
    adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
  else
    error('bad option opt.optim')
  end
  -- do a cnn update (if finetuning, and if rnn above us is not warming up right now)
  if funetune_cnn then
    if opt.cnn_optim == 'sgd' then
      sgd(cnn_params, cnn_grad_params, cnn_learning_rate)
    elseif opt.cnn_optim == 'sgdm' then
      sgdm(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, cnn_optim_state)
    elseif opt.cnn_optim == 'adam' then
      adam(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, cnn_optim_state)
    else
      error('bad option for opt.cnn_optim')
    end
  end
end


function train(iter)
  -- eval loss/gradient
  local finetune = false
  if opt.finetune_cnn_after >= 0 and iter > opt.finetune_cnn_after then
    finetune = true
  end
  local losses = lossFun(finetune)
  local learning_rate, cnn_learning_rate = lr_policy(iter)

  --[[
  print(string.format(
    'claming %f%% of gradients', 
    100*torch.mean(torch.gt(torch.abs(grad_params), opt.grad_clip))
  ))
  --]]
  -- clip gradients
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  -- apply L2 regularization
  if finetune and opt.cnn_weight_decay > 0 then
    cnn_grad_params:add(opt.cnn_weight_decay, cnn_params)
    -- note: we don't bother adding the l2 loss to the total loss, meh.
    cnn_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end
  update_grad(finetune, learning_rate, cnn_learning_rate)

  return losses, finetune
end


function check_termination(iter, loss0, losses)
  local exit_flag = false
  -- good idea to do this once in a while, i think
  if iter % 10 == 0 then 
    collectgarbage() 
  end
  if loss0 == nil then 
    loss0 = losses.total_loss 
  end
  if losses.total_loss > loss0 * 20 then
    print('loss seems to be exploding, quitting.')
    exit_flag = true
  end
  -- stopping criterion
  if opt.max_iters > 0 and iter >= opt.max_iters then 
    exit_flag = true
  end

  return exit_flag
end
