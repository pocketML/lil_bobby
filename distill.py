from common import argparsers
from compression.distillation import data
from compression.distillation.models import TangBILSTM, TangLoss, load_teacher
import torch.nn as nn

def main(args, sacred_experiment=None):
    print("Sit back, tighten your seat belt, and prepare for the ride of your life 🚀")

    device = 'cpu' if args.cpu else 'cuda:0'
    if args.generate_loss:
        data.generate_distillation_loss(args)
    if args.play:
        task = args.task
        teacher = load_teacher(task, args.cpu)
        model = TangBILSTM(teacher.task.label_dictionary)
        mse = nn.MSELoss()
        ce = nn.CrossEntropyLoss()
        
        sents = [["hej, hej", "what what"]]
        data = model.encode(sents)
        print(data)
        exit(0)

        model.to(device)
        mse.to(device)
        ce.to(device)

        criterion = TangLoss(0.5, mse, ce)
        distillation_data = data.load_distillation_data(task)
        val_data = data.load_val_data(task)
        dataloaders = data.get_dataloaders(50, *distillation_data, *val_data)

        for epoch in range(10):
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                
                running_loss = 0.0
                running_corrects = 0.0



        #encoded = model.encode([sent1, sent2])
        #out = model(encoded)
        #print(out)
        #print(out.size())

if __name__ == "__main__":
    ARGS = argparsers.args_distill()

    main(ARGS)

# -5.6615e-02, -1.4662e-01,  9.1819e-02,  8.9149e-02, -1.0267e-02