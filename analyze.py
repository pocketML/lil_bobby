from fairseq.models.roberta import RobertaModel
from examples.roberta import commonsense_qa # load the Commonsense QA task
from analysis import parameters
from compression import pruning

roberta = RobertaModel.from_pretrained('./models/checkpoints', 'checkpoint_best.pt', 'data/CommonsenseQA')
roberta.eval() # disable dropout

# weight_histogram_for_all_transformers(roberta)
parameters.print_threshold_stats(roberta)
pruning.magnitude(roberta, 0.05)
parameters.print_threshold_stats(roberta)
#parameters.print_model_size(roberta)