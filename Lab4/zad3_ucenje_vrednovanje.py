import time
import torch.optim
from zad1_ucitavanje_podataka import MNISTMetricDataset
from torch.utils.data import DataLoader
from zad2_model_metricko_ugradivanje import SimpleMetricEmbedding
from utils import train, evaluate, compute_representations
import sys
from torch.utils.data import Subset

EVAL_ON_TEST = True
EVAL_ON_TRAIN = False

ZAD3B_SUBSET = False
ZAD3C_IDENTITY_MODEL = False
ZAD3E_REMOVE_CLASS = None  #None ili broj klase


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")

    # CHANGE ACCORDING TO YOUR PREFERENCE
    mnist_download_root = "./mnist/"
    ds_train = MNISTMetricDataset(mnist_download_root, split='train')
    ds_test = MNISTMetricDataset(mnist_download_root, split='test')
    ds_traineval = MNISTMetricDataset(mnist_download_root, split='traineval')

    model_name = "model_crb_margin1"

    
    num_classes = 10

    print(f"> Loaded {len(ds_train)} training images!")
    print(f"> Loaded {len(ds_test)} validation images!")

    #ZAD 3.b - uzimamo podskup train i validation skupa
    if ZAD3B_SUBSET:
        subset_indices = torch.randperm(len(ds_train))[:1000]
        ds_train = Subset(ds_train, subset_indices)
        subset_indices = torch.randperm(len(ds_traineval))[:500]
        ds_traineval = Subset(ds_traineval, subset_indices)
        EVAL_ON_TEST = False
        EVAL_ON_TRAIN = True
        model_name="model_subset"


    train_loader = DataLoader(
        ds_train,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    traineval_loader = DataLoader(
        ds_traineval,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    emb_size = 32
    model = SimpleMetricEmbedding(1, emb_size).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    if ZAD3E_REMOVE_CLASS is not None:

        #TRAINING SET
        ds_train_remove_class = MNISTMetricDataset(mnist_download_root, split='train', remove_class=ZAD3E_REMOVE_CLASS)
        subset_indices = torch.randperm(len(ds_train_remove_class))[:1000]
        ds_train_remove_class = Subset(ds_train_remove_class, subset_indices)

        train_remove_class_loader = DataLoader(
            ds_train_remove_class,
            batch_size=64,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            drop_last=True
        )

        #VALIDATION SET
        subset_indices = torch.randperm(len(ds_traineval))[:500]
        ds_traineval = Subset(ds_traineval, subset_indices)

        traineval_loader = DataLoader(
            ds_traineval,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=1
        )   

        epochs = 3
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            t0 = time.perf_counter()
            train_loss = train(model, optimizer, train_remove_class_loader, device, model_name=f"model_remove_class_{ZAD3E_REMOVE_CLASS}")
            print(f"Mean Loss in Epoch {epoch}: {train_loss:.3f}")

            print("Computing mean representations for evaluation...")
            representations = compute_representations(model, train_loader, num_classes, emb_size, device)
            
            print("Evaluating on training set...")
            acc1 = evaluate(model, representations, traineval_loader, device)
            print(f"Epoch {epoch}: Train Top1 Acc: {round(acc1 * 100, 2)}%")

            t1 = time.perf_counter()
            print(f"Epoch time (sec): {(t1-t0):.1f}")
        sys.exit(0)


    if ZAD3C_IDENTITY_MODEL:
        from zad3c_IdentityModel import IdentityModel
        emb_size = 32
        model = IdentityModel().to(device)
        #ne radimo train, radimo samo validation
        representations = compute_representations(model, train_loader, num_classes, 1*28*28, device)
        acc1 = evaluate(model, representations, traineval_loader, device)
        print(f"Traineval Top1 Acc: {round(acc1 * 100, 2)}%")
        sys.exit(0)


    epochs = 3
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        t0 = time.perf_counter()
        train_loss = train(model, optimizer, train_loader, device, model_name=f"{model_name}")
        print(f"Mean Loss in Epoch {epoch}: {train_loss:.3f}")
        if EVAL_ON_TEST or EVAL_ON_TRAIN:
            print("Computing mean representations for evaluation...")
            representations = compute_representations(model, train_loader, num_classes, emb_size, device)
        if EVAL_ON_TRAIN:
            print("Evaluating on training set...")
            acc1 = evaluate(model, representations, traineval_loader, device)
            print(f"Epoch {epoch}: Train Top1 Acc: {round(acc1 * 100, 2)}%")
        if EVAL_ON_TEST:
            print("Evaluating on test set...")
            acc1 = evaluate(model, representations, test_loader, device)
            print(f"Epoch {epoch}: Test Accuracy: {acc1 * 100:.2f}%")
        t1 = time.perf_counter()
        print(f"Epoch time (sec): {(t1-t0):.1f}")