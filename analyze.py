from fairseq.models.roberta import RobertaModel
from examples.roberta import commonsense_qa # load the Commonsense QA task
from analysis import parameters
import argparsers

# weight_histogram_for_all_transformers(roberta)
#parameters.print_threshold_stats(roberta)
#parameters.print_threshold_stats(roberta)
#parameters.print_model_size(roberta)

if __name__ == "__main__":
    #roberta = RobertaModel.from_pretrained('./models/checkpoints', 'checkpoint_best.pt', 'data/CommonsenseQA')
    #roberta.eval() # disable dropout
    args = argparsers.args_analyze()
    print(args)
    print(args.model_path)
    #args = parser.parse_args()
