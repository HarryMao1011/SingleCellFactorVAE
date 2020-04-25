## This file is for training FactorVAE under scVAE
from FactorVAE import *

gene_dataset = CortexDataset(save_path=save_path, total_genes=None)
gene_dataset.subsample_genes(1000, mode="variance")
gene_dataset.make_gene_names_lower()

n_epochs = 400
lr = 1e-3
use_cuda = False
## setting up the training model
# vae = VAE(gene_dataset.nb_genes)
# trainer = UnsupervisedTrainer(
#     vae,
#     gene_dataset,
#     train_size=0.90,
#     use_cuda=use_cuda,
#     frequency=5,
# )
# trainer.train(n_epochs=n_epochs, lr=lr)


factorVAE = FactorVAE(gene_dataset.nb_genes)
# n_epochs = 2
# lr = 1e-3
# use_cuda = False
factortrainer = factorTrain(
    factorVAE,
    gene_dataset,
    train_size=0.90,
    use_cuda=use_cuda,
    frequency=5,
)
factortrainer.train(n_epochs=n_epochs, lr=lr)