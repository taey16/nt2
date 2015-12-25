
-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', false)
  local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

  protos.cnn:evaluate()
  protos.lm:evaluate()
  -- rewind iteator back to first datapoint in the split
  loader:resetIterator(split)
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local predictions = {}
  local vocab = loader:getVocab()

  while true do
    -- fetch a batch of data
    local data = loader:getBatch({
       batch_size = opt.batch_size, 
       image_size = opt.image_size, 
       split = split, 
       seq_per_img = opt.seq_per_img
    })
    if opt.use_vgg then
      data.images = 
        -- preprocess in place, and don't augment
        net_utils.prepro(data.images, opt.crop_size, false, opt.gpuid >= 0)
    else
      data.images = 
        net_utils.preprocess_inception7(data.images, opt.crop_size, false, opt.gpuid >= 0)
    end
    n = n + data.images:size(1)

    -- forward the model to get loss
    local feats = protos.cnn:forward(data.images)
    local expanded_feats = protos.expander:forward(feats)
    local logprobs = protos.lm:forward{expanded_feats, data.labels}
    local loss = protos.crit:forward(logprobs, data.labels)
    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1

    -- forward the model to also get generated samples for each image
    local seq = protos.lm:sample(feats)
    local sents = net_utils.decode_sequence(vocab, seq)
    for k=1,#sents do
      local entry = {image_id = data.infos[k].id, caption = sents[k]}
      table.insert(predictions, entry)
      if verbose then
        print(string.format('image_id %s: %s', entry.image_id, entry.caption))
      end
    end

    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, val_images_use)
    if verbose then
      print(string.format(
        'evaluating validation performance... %d/%d (%f)', ix0-1, ix1, loss)
      )
    end

    if loss_evals % 10 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if n >= val_images_use then break end -- we've used enough images
  end

  local lang_stats
  if opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(predictions, opt.id)
  end

  return loss_sum/loss_evals, predictions, lang_stats
end

