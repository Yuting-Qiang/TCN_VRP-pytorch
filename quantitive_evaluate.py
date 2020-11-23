import numpy as np
import os
import sys
from settings import params, fix_settings

def evaluate_relationship_prediction(predictions, targets):
    total = 0.
    res = {'recall@20': 0., 'recall@50': 0., 'recall@100': 0.}
    for i, prediction in enumerate(predictions):
        # ensure not replicate relationships
        prediction = [tuple((int(x[0]), int(x[1]), int(x[2]))) for j, x in enumerate(prediction)]
        target = [tuple(x) for j, x in enumerate(targets[i])]
        print(i)
        # print('prediction', prediction)
        # print('target', target)
        assert len(list(set(prediction))) == len(prediction)
        assert len(list(set(target))) == len(target)
        total += len(target)
        res['recall@20'] += len(set.intersection(set(prediction[:20]), set(target)))
        res['recall@50'] += len(set.intersection(set(prediction[:50]), set(target)))
        res['recall@100'] += len(set.intersection(set(prediction[:100]), set(target)))
    res['recall@20'] /= total
    res['recall@50'] /= total
    res['recall@100'] /= total
    return res


def evaluate_fewshot_relationship_prediction(predictions, targets, few_shot):
    annotations = pickle.load(open(os.path.join('data', params['dataset'], 'annotations.pkl'), 'rb'))
    train_annotations = annotations['train_annotations']
    test_annotations = annotations['test_annotations']
    train_global_tensor = pickle.load(open(os.path.join(root_path, 'train_global_tensor.pkl'), 'rb'))
    train_global_tensor *= len(train_annotations)
    test_global_tensor = pickle.load(open(os.path.join(root_path, 'test_global_tensor.pkl'), 'rb'))
    test_global_tensor *= len(test_annotations)
    global_tensor = train_global_tensor+test_global_tensor
    few_shot_index = np.where((global_tensor>0) & (train_global_tensor <=1))
    few_shot_index_set = set([(few_shot_index[0][i], few_shot_index[1][i], few_shot_index[2][i]) for i in range(len(few_shot_index[0]))])
    
    total = 0.
    res = {'recall@50': 0., 'recall@100': 0.}
    for i, prediction in enumerate(predictions):
        # ensure not replicate relationships
        prediction = [tuple((int(x[0]), int(x[1]), int(x[2]))) for j, x in enumerate(prediction)]
        target = [tuple(x) for j, x in enumerate(targets[i])]
        assert len(list(set(prediction))) == len(prediction)
        assert len(list(set(target))) == len(target)

        target_set = set.intersection(few_shot_index_set, target)
        print(i)
        # print('prediction', prediction)
        # print('target', target)
        total += len(target_set)
        res['recall@50'] += len(set.intersection(set(prediction[:50]), target_set)))
        res['recall@100'] += len(set.intersection(set(prediction[:100]), target_set)))

    res['recall@50'] /= total
    res['recall@100'] /= total
    return res

def main(params, which_expr):
    # load predictions
    predictions = np.load(os.path.join('results', params['dataset'], 'predictions.npy'), allow_pickle=True)
    targets = np.load(os.path.join('results', params['dataset'], 'targets.npy'), allow_pickle=True)
    if which_expr == 'relpred' or which_expr == 'all': 
        res = evaluate_relationship_prediction(predictions, targets)
    if which_expr == 'fewshot':
        res = evalutate_relationship_prediction(predictions, sys.argv[2])
    return res

if __name__ == '__main__':
    from settings import params, fix_settings
    fix_settings()
    main(params, sys.argv[1])
