
require 'models.LanguageModel'

local protos = {}

if string.len(opt.start_from) > 0 then
  -- load protos from file
  print('initializing weights from ' .. opt.start_from)
  local loaded_checkpoint = torch.load(opt.start_from)
  protos = loaded_checkpoint.protos
  net_utils.unsanitize_gradients(protos.cnn)
  local lm_modules = protos.lm:getModulesList()
  for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end
  protos.crit = nn.LanguageModelCriterion() -- not in checkpoints, create manually
  protos.expander = nn.FeatExpander(opt.seq_per_img) -- not in checkpoints, create manually
else

  -- create protos from scratch
  -- intialize language model
  local lmOpt = {}
  lmOpt.vocab_size = loader:getVocabSize()
  lmOpt.input_encoding_size = opt.input_encoding_size
  lmOpt.rnn_size = opt.rnn_size
  lmOpt.num_layers = 1
  lmOpt.dropout = opt.drop_prob_lm
  lmOpt.seq_length = loader:getSeqLength()
  lmOpt.batch_size = opt.batch_size * opt.seq_per_img
  protos.lm = nn.LanguageModel(lmOpt)

  -- initialize the ConvNet
  -- override to nn if gpu is disabled
  if opt.gpuid == -1 then cnn_backend = 'nn' end
  if opt.use_vgg then
    require 'loadcaffe'
    local cnn_backend = opt.backend
    local cnn_raw = 
      loadcaffe.load(opt.cnn_proto, opt.cnn_model, cnn_backend)
    protos.cnn = 
      net_utils.build_cnn(cnn_raw, {encoding_size = opt.input_encoding_size, backend = cnn_backend})
  else
    protos.cnn = 
      net_utils.build_inception_cnn({encoding_size = opt.input_encoding_size, model_filename = opt.torch_model})
  end

  -- initialize a special FeatExpander module that "corrects" for the batch number discrepancy 
  -- where we have multiple captions per one image in a batch. This is done for efficiency
  -- because doing a CNN forward pass is expensive. We expand out the CNN features for each sentence
  protos.expander = nn.FeatExpander(opt.seq_per_img)

  -- criterion for the language model
  protos.crit = nn.LanguageModelCriterion()
end


-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end


-- construct thin module clones that share parameters with the actual
-- modules. These thin module will have no intermediates and will be used
-- for checkpointing to write significantly smaller checkpoint files
local thin_lm = protos.lm:clone()
-- TODO: we are assuming that LM has specific members! figure out clean way to get rid of, not modular.
thin_lm.core:share(protos.lm.core, 'weight', 'bias')
thin_lm.lookup_table:share(protos.lm.lookup_table, 'weight', 'bias')
local thin_cnn = protos.cnn:clone('weight', 'bias')
-- sanitize all modules of gradient storage so that we dont save big checkpoints
net_utils.sanitize_gradients(thin_cnn)
local lm_modules = thin_lm:getModulesList()
for k,v in pairs(lm_modules) do net_utils.sanitize_gradients(v) end

-- create clones and ensure parameter sharing. we have to do this 
-- all the way here at the end because calls such as :cuda() and
-- :getParameters() reshuffle memory around.
protos.lm:createClones()
collectgarbage() -- "yeah, sure why not"

return protos, thin_lm, thin_cnn

