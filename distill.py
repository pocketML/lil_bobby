from common import argparsers
from compression.distillation import loss_data, student_models

def main(args, sacred_experiment=None):
    print("Sit back, tighten your seat belt, and prepare for the ride of your life 🚀")

    if args.generate_loss:
        loss_data.generate(args)
    if args.play:
        model = student_models.TangBILSTM()
        sent1 = "Herllo good friendy friend"
        sent2 = "How are you doing today my good friend."
        encoded = model.encode([sent1, sent2])
        out = model(encoded)
        #print(out)
        #print(out.size())

if __name__ == "__main__":
    ARGS = argparsers.args_distill()

    main(ARGS)

# -5.6615e-02, -1.4662e-01,  9.1819e-02,  8.9149e-02, -1.0267e-02