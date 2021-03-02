from fairseq.models.roberta import RobertaModel
from examples.roberta import commonsense_qa # load the Commonsense QA task
from analysis import parameters
from common import argparsers

# weight_histogram_for_all_transformers(roberta)
#parameters.print_threshold_stats(roberta)
#parameters.print_threshold_stats(roberta)
#parameters.print_model_size(roberta)

if __name__ == "__main__":
    #roberta = RobertaModel.from_pretrained('./models/checkpoints', 'checkpoint_best.pt', 'data/CommonsenseQA')
    #roberta.eval() # disable dropout
    args = argparsers.args_analyze()
    print(args.model)
    from download import TASK_DATASET_PATHS, MODEL_PATHS

    base_models = [k for k in MODEL_PATHS.keys()]
    finetuned = [k for k in TASK_DATASET_PATHS.keys()]

    if args.model in base_models:
        model_path = 'models/roberta.' + args.model
        model = 
    elif args.model in finetuned:
        model_path = ''

    if args.model_size:
        parameters.print_model_size()
    #args = parser.parse_args()
