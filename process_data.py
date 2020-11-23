import json
import os
import pickle
import sys
import h5py
import numpy as np
# from PIL import Image

def construct_annotations(ann, n_entities, n_predicates):
    annotations = []
    for key, val in ann.items():
        item = {}
        item['img'] = key
        item['relations'] = []
        for x in val:
            assert x['subject']['category'] < n_entities
            assert x['object']['category'] < n_entities
            assert x['predicate'] < n_predicates
            item['relations'].append([x['subject']['category'], x['predicate'], x['object']['category']])
        annotations.append(item)
    return annotations

def construct_global_tensor(annotations, n_entities, n_predicates):
    global_tensor = np.zeros([n_entities, n_entities, n_predicates], dtype=np.float64)
    for i, item in enumerate(annotations):
        relation_set = set() # count each relation once for every image
        for j, rel in enumerate(annotations[i]['relations']):
            if (rel[0], rel[2], rel[1]) not in relation_set:
                global_tensor[rel[0], rel[2], rel[1]] += 1
                relation_set.add((rel[0], rel[2], rel[1]))
    return global_tensor/len(annotations)

def construct_label_weights(global_tensor):
    total = np.sum(global_tensor)
    mask = np.where(global_tensor > 0)
    label_weights = np.zeros(shape=global_tensor.shape)
    label_weights[mask[0], mask[1], mask[2]] = -np.log(global_tensor[mask[0], mask[1], mask[2]]/total)
    return label_weights

def construct_sample_weights(label_weights, annotations):
    sample_weight = np.zeros(len(annotations))
    for i, item in enumerate(annotations):
        for j, rel in enumerate(annotations[i]['relations']):
            sample_weight[i] += label_weights[rel[0], rel[2], rel[1]]
    return sample_weight

def proc_data(dataset):
    root_path = os.path.join('data', dataset)
    if dataset == 'vrd':
        # load data
        ann_train = json.load(open(os.path.join(root_path, 'annotations_train.json'), 'r'))
        ann_test = json.load(open(os.path.join(root_path, 'annotations_test.json'), 'r'))
        objects = json.load(open(os.path.join(root_path, 'objects.json'), 'r'))
        predicates = json.load(open(os.path.join(root_path, 'predicates.json'), 'r'))

        # prepare annotations
        train_annotations = construct_annotations(ann_train, len(objects), len(predicates))
        test_annotations = construct_annotations(ann_test, len(objects), len(predicates))
    elif dataset == 'vg200':
        # load data
        root_path = os.path.join('data', 'vg200')
        metadata = h5py.File(os.path.join(root_path, 'vg1_2_meta.h5'), 'r')
        idx2objects = metadata['meta']['cls']['idx2name']
        objects = [idx2objects[str(idx)][()] for idx in range(1, len(list(idx2objects.keys())))] # 0 is background
        idx2predicates = metadata['meta']['pre']['idx2name']
        predicates = [idx2predicates[str(key)][()] for key in range(len(list(idx2predicates.keys())))]
        imid2path = metadata['meta']['imid2path']
        train_data = metadata['gt']['train']
        test_data = metadata['gt']['test']
        train_annotations = construct_vg200_annotations(train_data, imid2path, len(list(idx2objects)), len(list(idx2predicates)))
        test_annotations = construct_vg200_annotations(test_data, imid2path, len(list(idx2objects)), len(idx2predicates))
    else:
        sys.exit(1)
    all_data = {'objects': objects, 'predicates': predicates,
                'train_annotations': train_annotations, 'test_annotations': test_annotations }
    pickle.dump(all_data, open(os.path.join(root_path,'annotations.pkl'), 'wb'))

    # prepared global tensors
    train_global_tensor = construct_global_tensor(train_annotations, len(objects), len(predicates))
    pickle.dump(train_global_tensor, open(os.path.join(root_path, 'train_global_tensor.pkl'), 'wb'))
    test_global_tensor = construct_global_tensor(test_annotations, len(objects), len(predicates))
    pickle.dump(test_global_tensor, open(os.path.join(root_path, 'test_global_tensor.pkl'), 'wb'))

    # prepare label weights
    label_weights = construct_label_weights(train_global_tensor)
    pickle.dump(label_weights, open(os.path.join(root_path, 'label_weights.pkl'), 'wb'))

    # prepare sample weights
    sample_weights = construct_sample_weights(label_weights, train_annotations)
    pickle.dump(sample_weights, open(os.path.join(root_path, 'sample_weights.pkl'), 'wb'))

def construct_vg200_annotations(ann, imid2path, n_entities, n_predicates):
    annotations = []
    for key, val in ann.items():
        item = {}
        item['img'] = os.path.basename(imid2path[key][()])
        item['relations'] = []
        for x in val['rlp_labels'][()]:
            sub_cls = x[0]-1
            pred_cls = x[1]
            obj_cls = x[2]-1
            assert sub_cls >= 0 and sub_cls < n_entities, print(sub_cls)
            assert obj_cls >= 0 and obj_cls < n_entities, print(obj_cls)
            assert pred_cls >= 0 and pred_cls < n_predicates, print(pred_cls)
            item['relations'].append([sub_cls, pred_cls, obj_cls])
        annotations.append(item)
    return annotations

if __name__ =='__main__':
    print(sys.argv)
    proc_data(sys.argv[1])
