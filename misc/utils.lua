local cjson = require 'cjson'
local utils = {}

-- Assume required if default_value is nil
function utils.getopt(opt, key, default_value)
  if default_value == nil and (opt == nil or opt[key] == nil) then
    error('error: required key ' .. key .. ' was not provided in an opt.')
  end
  if opt == nil then return default_value end
  local v = opt[key]
  if v == nil then v = default_value end
  return v
end

function get_val()
  local file = io.open '/storage/ImageNet/ILSVRC2012/val_synset.txt'
  local image_list ={}
  local label_list ={}
  local synset_list={}

  while true do
    local line = file:read()
    if not line then break end
    local item  =string.split(line, ' ')
    local synset=string.split(item[1], '/')
    table.insert(image_list, item[1])
    table.insert(label_list, item[2])
    table.insert(synset_list,synset[1])
  end
  file:close()
  return image_list, label_list, synset_list
end


function utils.read_text(path)
  local file = io.open(path, 'r')
  local image_list = {}
  local sentence_list = {}
  local roi_list = {}

  while true do
    local line = file:read()
    if not line then break end
    local item = string.split(line, ';;')
    local filename = item[1]
    local roi = string.split(item[2], ' ')
    local sentence = string.split(item[3], ' ')
    table.insert(image_list, filename)
    table.insert(roi_list, roi)
    table.insert(sentence_list, sentence)
  end

end
function utils.read_json(path)
  local file = io.open(path, 'r')
  local text = file:read()
  file:close()
  local info = cjson.decode(text)
  return info
end

function utils.write_json(path, j)
  -- API reference http://www.kyne.com.au/~mark/software/lua-cjson-manual.html#encode
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  assert(file, 'check path: '..path)
  file:write(text)
  file:close()
end

-- dicts is a list of tables of k:v pairs, create a single
-- k:v table that has the mean of the v's for each k
-- assumes that all dicts have same keys always
function utils.dict_average(dicts)
  local dict = {}
  local n = 0
  for i,d in pairs(dicts) do
    for k,v in pairs(d) do
      if dict[k] == nil then dict[k] = 0 end
      dict[k] = dict[k] + v
    end
    n=n+1
  end
  for k,v in pairs(dict) do
    dict[k] = dict[k] / n -- produce the average
  end
  return dict
end

-- seriously this is kind of ridiculous
function utils.count_keys(t)
  local n = 0
  for k,v in pairs(t) do
    n = n + 1
  end
  return n
end

-- return average of all values in a table...
function utils.average_values(t)
  local n = 0
  local vsum = 0
  for k,v in pairs(t) do
    vsum = vsum + v
    n = n + 1
  end
  return vsum / n
end

return utils
