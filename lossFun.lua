
function lossFun(finetune_cnn)
  protos.cnn:training()
  protos.lm:training()
  grad_params:zero()
  if finetune_cnn then
    cnn_grad_params:zero()
  end

  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data  
  local data = 
    loader:getBatch{
      batch_size = opt.batch_size, 
      image_size = opt.image_size, 
      split = 'train', 
      seq_per_img = opt.seq_per_img
    }
  if opt.use_vgg then
    -- preproces in-place, data augment
    data.images = 
      net_utils.prepro(data.images, opt.crop_size, true, opt.gpuid >= 0)
  else
    -- preproces in-place, data augment
    data.images = 
      net_utils.preprocess_inception7(data.images, opt.crop_size, true, opt.gpuid >= 0)
  end
  -- data.images: Nx3x224x224 
  -- data.seq: LxM where L is sequence length upper bound, and M = N*seq_per_img

  -- forward the ConvNet on images (most work happens here)
  --if not data.images:isContiguous() then data.images = data.images:clone() end
  local feats = protos.cnn:forward(data.images)
  -- we have to expand out image features, once for each sentence
  local expanded_feats = protos.expander:forward(feats)
  -- forward the language model
  local logprobs = protos.lm:forward{expanded_feats, data.labels}
  -- forward the language model criterion
  local loss = protos.crit:forward(logprobs, data.labels)
  
  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dlogprobs = protos.crit:backward(logprobs, data.labels)
  -- backprop language model
  local dexpanded_feats, ddummy = 
    unpack(protos.lm:backward({expanded_feats, data.labels}, dlogprobs))
  -- backprop the CNN, but only if we are finetuning
  if finetune_cnn then
    local dfeats = protos.expander:backward(feats, dexpanded_feats)
    local dx = protos.cnn:backward(data.images, dfeats)
  end

  -- and lets get out!
  local losses = { total_loss = loss }
  return losses
end


