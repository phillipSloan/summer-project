def get_paths(mask):
    data_dir = os.path.join(os.getcwd(), 'data/train')
    tmp = []
    for path in Path(data_dir).rglob(mask):
        tmp.append(path.resolve())
    return tmp

def load_metrics(filename):
    import pickle
    with open(filename, "rb") as f:
        saved_list = pickle.load(f)
    return saved_list

def save_metrics(filename, metrics):
    import pickle
    lists_to_save = metrics
    with open(filename,"wb") as f:
        pickle.dump(lists_to_save, f)
        
def rand_crop(images, labels, cropper):
    cropped_inputs = []
    cropped_labels = []
    assert images.shape[0] == labels.shape[0]
    for i in range(images.shape[0]):
        pair =  {"image": images[i], "label":labels[i]}
        out = cropper(pair)
        for i in range(len(out)):
            cropped_inputs.append(out[i]['image'])
            cropped_labels.append(out[i]['label'])
    imgs = torch.stack(cropped_inputs)
    lbls = torch.stack(cropped_labels)
    return imgs, lbls