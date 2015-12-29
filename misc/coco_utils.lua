

function get_val(filename_)
  local filename = filename_ or '/storage/coco/test2015/'
  local file = io.open filename
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
  return image_list, label_list, synset_list
end

