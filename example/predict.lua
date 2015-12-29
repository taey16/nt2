require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn'
local cjson = require 'cjson'
package.path = '../?.lua;' .. package.path
require 'misc.DataLoaderRaw'
require 'models.LanguageModel'
local net_utils = require 'misc.net_utils'


local model_filename = 
  '/storage/coco/checkpoints/_inception7_bs16_encode256_layer2/model_id_inception7_bs16_encode256_layer2.t7'
  --'/storage/coco/checkpoints/_inception7_bs16_encode512/model_id_inception7_bs16_encode512.t7'
  --'/storage/coco/checkpoints/_inception7_bs16_encode512_finetune_lr4e-6_clr1e-7_wc2e-5/model_id_inception7_bs16_encode512_finetune_lr4e-6_clr1e-7_wc2e-5.t7'
local checkpoint = torch.load(model_filename)
local batch_size = checkpoint.opt.batch_size
local opt = {
  'rnn_size', 
  'input_encoding_size', 
  'drop_prob_lm', 
  'cnn_proto', 
  'cnn_model', 
  'seq_per_img', 
  'image_size', 
  'crop_size'
}
for k,v in pairs(opt) do 
  opt[v] = checkpoint.opt[v]
end
local vocab = checkpoint.vocab
local sample_opts = { 
  sample_max = opt.sample_max, 
  beam_size = 4, 
  temperature = opt.temperature 
}

local protos = checkpoint.protos
protos.expander = nn.FeatExpander(opt.seq_per_img)
protos.lm:createClones()
for k,v in pairs(protos) do v:cuda() end
protos.cnn:evaluate()
protos.lm:evaluate()

local json_filename = 
  '/storage/coco/coco_raw.json'
local file = io.open(json_filename, 'r')
local text = file:read()
file:close()
local info = cjson.decode(text)

local output_dic = {}
local outfile = io.open(
  "model_id_inception7_bs16_encode256_layer2.t7.html", "w"
  --"model_id_inception7_bs16_encode512.t7.html", "w"
  --"model_id_inception7_bs16_encode512_finetune_lr4e-6_clr1e-7_wc2e-5.t7.html", "w"
)
io.output(outfile)
io.write("<html>\n  <head>\n    <table>\n      <tr>\n")

--[[
for k, v in pairs(info) do print(k) end
annotations
images
info
licenses
--]]
--[[
for k, v in pairs(info['images']) do print(info['images'][k]) end
{
  file_name : "COCO_val2014_000000560744.jpg"
  coco_url : "http://mscoco.org/images/560744"
  flickr_url : "http://farm4.staticflickr.com/3791/9109408773_2235e8972e_z.jpg"
  id : 560744
  height : 480
  license : 1
  date_captured : "2013-11-25 15:04:29"
  width : 640
}
--]]

function permute(tab, n, count)
  n = n or #tab
  for i = 1, count or n do
    local j = math.random(i, n)
    tab[i], tab[j] = tab[j], tab[i]
  end
  return tab
end
info = permute(info)

local iter = 1
for k, v in ipairs(info) do
  local fname = v['file_path'] 
  local captions = v['captions']
  print(fname)
  local url = string.gsub(fname, '/storage', 'http://10.202.35.109:2596/PBrain')
  io.write(string.format("        <td><img src=\"%s\" height=\"292\" width=\"292\"></br>\n", url))
  for k,sent  in ipairs(captions) do
    io.write(string.format("          <font color=\"blue\">%s</font></br>\n", sent))
    print(sent)
  end

  local img = image.load(fname)
  img = image.scale(img, opt.image_size, opt.image_size)
  if img:size(1) == 1 then
    img = img:view(1,img:size(2), img:size(3)):repeatTensor(3,1,1)
  end
  img = net_utils.preprocess_inception7_predict(img, opt.crop_size, false, 1)
  local data = torch.CudaTensor(2, 3, opt.crop_size, opt.crop_size):fill(0)
  data[{{1},{},{},{}}] = img
  local feats = protos.cnn:forward(data)
  local seq = protos.lm:sample(feats, sample_opts)
  local sents = net_utils.decode_sequence(vocab, seq)
  output_dic['url'] = sents
  json_obj = cjson.encode(output_dic)
  print('\n')
  io.write(string.format("        <font color=\"green\">%s</font>", sents[1]))
  io.write("      </td>\n")
  if iter % 5 == 0 then
    io.write("    </tr>\n<tr>\n")
  end
  for k,_ in pairs(sents) do
    print(sents[k])
  end
  print('\n')
  iter = iter+1
end

io.write("    </table>\n  </head>\n</html>")
io.close(outfile)
--]]
