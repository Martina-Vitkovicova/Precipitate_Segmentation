import denoiseg.image_utils as iu
import pathlib

def read_image_prefer_type(
    root, 
    stem_name, 
    ext_preference = ['tif','tiff','png','jpg']
):
    ext_preference = [e.strip('x') for e in ext_preference]
    root = pathlib.Path(root)
    paths = list(root.rglob(f'{stem_name}.*'))
    path_dict = _construct_path_dict(paths)
    return _filter_paths(path_dict,ext_preference)

def _construct_path_dict(imgs_paths):
    uniq_paths = {}
    for p in imgs_paths:
        key = str(p.parent)
        val = uniq_paths.setdefault(key,{})
        val[p.suffix.strip('.')] = p
    return uniq_paths

def _filter_paths(path_dict, ext_preference,raise_on_not_found = False):
    def _filter(v,exts):
        for ext in exts:
            if ext in v:
                return v[ext]
        
        msg = f"No image of types {ext_preference} found in paths {v.values()}"
        if raise_on_not_found:
            raise Exception(msg)
        else:
            logger.warning(msg)
        
    only_best_fit = {k:_filter(v,ext_preference) for k,v in path_dict.items()}
    return list(only_best_fit.values())

def load_pairs(labeled_root, include_paths = False):
    imgs_paths = read_image_prefer_type(labeled_root,'img')
    imgs_lbl = [iu.load_image(img_path) for img_path in imgs_paths]

    mask_paths = [ img_path.parent/"mask.png" for img_path in imgs_paths]
    gts_lbl = [iu.load_image(img_path) for img_path in mask_paths]
    
    if include_paths:
        return imgs_lbl,gts_lbl, imgs_paths,mask_paths
    else:
        return imgs_lbl,gts_lbl