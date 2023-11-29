from param_parser import parameter_parser
import os


args = parameter_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

from utils import *
from train_eval import *
args = config_args(args)
print(args)
if args.dataset_name in ["IMDB-BINARY", "REDDIT-BINARY", "COLLAB", "IMDB-MULTI"]:
    dataset = get_dataset(args.dataset_name)
    num_node_features = dataset.num_node_features
    num_classes = dataset.num_classes
print("num_node_features : %d, num_classes : %d"%(num_node_features, num_classes))

if args.model == 'GCN':
    model = GCN(args, num_node_features, num_classes, args.hidden_dim)
elif args.model == 'GIN':
    model = GIN(args, num_node_features, num_classes, args.hidden_dim)
elif args.model == 'VIB':
    model = VIBGSL(args, num_node_features, num_classes)
print(model.__repr__())
cross_validation_with_val_set(dataset, model, args.folds, args.epochs, args.batch_size, args.test_batch_size,
                              args.lr, args.lr_decay_factor, args.lr_decay_step_size,
                              args.weight_decay, logger=None, args=args)