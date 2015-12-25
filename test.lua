
paths.dofile('eval_split.lua')

function conditional_save(iter, val_loss, val_predictions, lang_stats)
  local checkpoint_path = path.join(opt.checkpoint_path, 'model_id' .. opt.id)
  -- write a (thin) json report
  local checkpoint = {}
  checkpoint.opt = opt
  checkpoint.iter = iter
  checkpoint.loss_history = loss_history
  checkpoint.val_loss_history = val_loss_history
  -- save these too for CIDEr/METEOR/etc eval
  checkpoint.val_predictions = val_predictions
  checkpoint.val_lang_stats_history = val_lang_stats_history
  utils.write_json(checkpoint_path .. '.json', checkpoint)
  print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

  -- write the full model checkpoint as well if we did better than ever
  local current_score
  if lang_stats then
    -- use CIDEr score for deciding how well we did
    current_score = lang_stats['CIDEr']
  else
    -- use the (negative) validation loss as a score
    current_score = -val_loss
  end
  if best_score == nil or current_score > best_score then
    best_score = current_score
    -- dont save on very first iteration
    if iter > 0 then
      -- include the protos (which have weights) and save to file
      local save_protos = {}
      -- these are shared clones, and point to correct param storage
      save_protos.lm = thin_lm
      save_protos.cnn = thin_cnn
      checkpoint.protos = save_protos
      -- also include the vocabulary mapping so that we can use the checkpoint 
      -- alone to run on arbitrary images without the data loader
      checkpoint.vocab = loader:getVocab()
      torch.save(checkpoint_path .. '.t7', checkpoint)
      print('wrote checkpoint to ' .. checkpoint_path .. '.t7')
    end
  end
end


function test(iter)
  -- evaluate the validation performance
  local val_loss, val_predictions, lang_stats = 
    eval_split('val', {val_images_use = opt.val_images_use})

  return val_loss, val_predictions, lang_stats
end
