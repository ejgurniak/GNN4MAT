import os
import shutil
import sys
from glob import glob

# from utilis import ROC_AUC_multiclass
import numpy as np
import torch
import torch.nn as nn
import yaml
import time
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.handlers.param_scheduler import LRScheduler
from ignite.metrics import Accuracy
from natsort import natsorted
from pymatgen.io.vasp.inputs import Poscar
from torch.optim.lr_scheduler import StepLR

from dataloader import get_train_val_test_loader
# from graph import CrystalGraphDataset, prepare_batch_fn

# from ignite.contrib.handlers.tensorboard_logger import *
from settings import Settings

# define the directory path
dir_path = "./validation_labels"

# Check if the directory exists
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"Directory '{dir_path}' created")
else:
    print(f"Directory '{dir_path}' already exists")

training_data = sys.argv[1]

try:
    checkpoint_dir = sys.argv[2]
except:
    checkpoint_dir = "model_checkpoints"

try:
    logfile = sys.argv[3]
except:
    logfile = "log.model"

try:
    current_model = sys.argv[4]
except:
    current_model = "GANN"
    print("NOTE: no model chosen, default to the Graph Attention Network by Banik et al")

print(f'current_model: {current_model}')

try:
    is_heterogeneous = sys.argv[5]
except:
    is_heterogeneous = "homogeneous"
    print("NOTE: heterogeneity not chosen, defaulting to homogeneous version")

if is_heterogeneous == "heterogeneous":
    print("heterogeneous graph chosen, input features will be modified based on Cu-Zr")
    from hetgraph import CrystalGraphDataset, prepare_batch_fn
else:
    print("homogeneous graph chosen, no modification of input features")
    from graph import CrystalGraphDataset, prepare_batch_fn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("next line is print(device) after init device")
print(device)

tic = 0.0
toc = 0.0


def train(
    graphs,
    settings,
    model,
    output_dir=os.getcwd() + "model",
    screen_log="log.model",
):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    global best_val_accuracy
    global total_batch_loss
    global iteration_count

    best_val_accuracy, total_batch_loss, iteration_count = -1e300, 0, 0

    train_loader, val_loader, test_loader = get_train_val_test_loader(
        graphs,
        collate_fn=graphs.collate,
        batch_size=settings.batch_size,
        train_ratio=settings.train_ratio,
        val_ratio=settings.val_ratio,
        test_ratio=settings.test_ratio,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
        train_size=settings.train_size,
        test_size=settings.test_size,
        val_size=settings.val_size,
    )

    val_dataset = val_loader.dataset
    # Assuming you have created the dataset instance named 'dataset'

    # Choose an index for the item you want to print
    idx = 0

    # Get the item from the dataset using the specified index
    item = val_dataset[idx]

    # Unpack the item tuple and print each element
    # print(f'length of item[0]: {len(item[0])}')
    bond_feature, angular_feature, nbr_idx, my_edge_index = item[0]
    target = item[1]
    val_dataset = val_loader.dataset

    # -----------losss--------------------

    def print_validation_details(model, val_loader, criterion, epoch):
        # out_file = open('validation_details.txt', 'a')
        # out_file.write('inside subroutine print_validation_details')
        val_labels_filename = f"./validation_labels/epoch_val_labels.{epoch}.txt"
        val_labels_file = open(val_labels_filename, 'w')
        predicted_labels = []
        val_indices = []
        model.eval()
        epoch_predicted_labels = []
        val_true_labels = []
        with torch.no_grad():
            for batch_idx, (bond_features, angular_features, species, nbr_idxs, crys_idxs, labels) in enumerate(val_loader):
                # batch_idx = batch_idx.to(device)
                bond_features = bond_features.to(device)
                nbr_idxs = nbr_idxs.to(device)
                angular_features = angular_features.to(device)
                crys_idxs = crys_idxs.to(device)
                labels = labels.to(device)
                species = species.to(device)
                data = bond_features, angular_features, species, nbr_idxs, crys_idxs

                outputs = model(data)
                _, predicted = torch.max(outputs, 1)

                start_idx = batch_idx * val_loader.batch_size
                end_idx = start_idx + labels.size(0)
                val_indices.extend(range(start_idx, end_idx))
                epoch_predicted_labels.extend(predicted.tolist())
                val_true_labels.extend(labels.tolist())

        # write val indices and labels to the val_labels_file
        val_labels_file.write('val_indices')
        val_labels_file.write(' ')
        val_labels_file.write('predicted_labels')
        val_labels_file.write(' ')
        val_labels_file.write('true_labels')
        val_labels_file.write("\n")
        for i in range(len(val_indices)):
            val_labels_file.write(str(val_indices[i]))
            val_labels_file.write(' ')
            val_labels_file.write(str(epoch_predicted_labels[i]))
            val_labels_file.write(' ')
            val_labels_file.write(str(val_true_labels[i]))
            val_labels_file.write("\n")
        val_labels_file.close()

        # out_file.close()

    # function to calculate validation loss
    def calculate_validation_loss(model, val_loader, criterion):
        model.eval() # Set the model to evaluation mode

        val_loss = 0.0
        with torch.no_grad():
            for bond_features, angular_features, species, nbr_idxs, crys_idxs, labels in val_loader: # only unpack inputs and labels
                data = bond_features, angular_features, species, nbr_idxs, crys_idxs
                bond_features = bond_features.to(device)
                nbr_idxs = nbr_idxs.to(device)
                angular_features = angular_features.to(device)
                crys_idxs = crys_idxs.to(device)
                labels = labels.to(device)
                species = species.to(device)
                data2 = bond_features, angular_features, species, nbr_idxs, crys_idxs

                outputs = model(data2) #HERE - try just one argument
                loss = criterion(outputs, labels)
                # Accumulate the loss across batches
                val_loss += loss.item() * settings.batch_size #CHECK HERE - may be incorrect
        # Compute the average loss across the validation set
        val_loss = val_loss / len(val_loader)

        model.train() # Set model back to training mode
        return val_loss
    # end function to calculate validation loss


    criterion = nn.CrossEntropyLoss()
    val_metrics = {
        # "rocauc": ROC_AUC_multiclass()
        "accuracy": Accuracy(),
    }

    # ---------model----------------

    model.to(device)

    if settings.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )

    if settings.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=settings.learning_rate,
            momentum=settings.momentum,
        )

    # ------------tariner---------------

    trainer = create_supervised_trainer(
        model,
        optimizer,
        criterion,
        prepare_batch=prepare_batch_fn,
        device=device,
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global total_batch_loss
        global iteration_count
        total_batch_loss += engine.state.output
        iteration_count += 1

    # ----------scheduler-------------------

    if settings.scheduler:

        torch_lr_scheduler = StepLR(
            optimizer, step_size=settings.step_size, gamma=settings.gamma
        )
        scheduler = LRScheduler(torch_lr_scheduler)
        trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)

        # @trainer.on(Events.EPOCH_STARTED)
        # def print_lr():
        #    print(optimizer.param_groups[0]["lr"])

    if settings.progress:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    # ----------Evaluator---------------------

    evaluator = create_supervised_evaluator(
        model,
        prepare_batch=prepare_batch_fn,
        metrics=val_metrics,
        device=device,
    )

    test_evaluator = create_supervised_evaluator(
        model,
        prepare_batch=prepare_batch_fn,
        metrics=val_metrics,
        device=device,
    )

    # ----------Checkpoint-------------------

    def Checkpoint(epoch, is_best, path=os.getcwd(), filename="checkpoint.pt"):

        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "trainer": trainer.state_dict(),
                "best_accuracy": best_val_accuracy,
            },
            "{}/{}_{}".format(path, filename, epoch),
        )

        if is_best:
            shutil.copyfile(
                "{}/{}_{}".format(path, filename, epoch),
                "{}/model_best.pt".format(path),
            )
            #EJG print statement:
            print("is_best trigerred at epoch: " + str(epoch)) #EJG

    # ----------------resume---------------------

    if settings.resume:

        if os.path.isfile("{}/checkpoint.pt".format(output_dir)):
            print(
                "=> loading checkpoint '{}'".format(
                    "{}/checkpoint.pt".format(output_dir)
                )
            )
            checkpoint = torch.load("{}/checkpoint.pt".format(output_dir))
            epoch = checkpoint["epoch"]
            settings.epochs = settings.epochs - epoch - 1
            best_val_error = checkpoint["best_accuracy"]
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["lr_scheduler"])
            trainer.load_state_dict(checkpoint["trainer"])

        else:
            print("=> no checkpoint file found")

    else:
        os.system("rm {}".format(screen_log))

    # -----------logging resulst after every epoch--------------

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):

        global best_val_accuracy
        global total_batch_loss
        global iteration_count

        # print("validating")
        epoch = engine.state.epoch

        avg_batch_loss = total_batch_loss / iteration_count
        total_batch_loss, iteration_count = 0, 0

        # print("validating")
        epoch = engine.state.epoch

        evaluator.run(val_loader)

        valmetrics = evaluator.state.metrics
        val_accuracy = valmetrics["accuracy"]
        val_loss = calculate_validation_loss(model, val_loader, criterion) 
        # print_validation_details(model, val_loader, criterion, epoch) #HERE

        if len(test_loader) != 0:
            test_evaluator.run(test_loader)
            testmetrics = test_evaluator.state.metrics
            test_accuracy = testmetrics["accuracy"]
            test_loss = test_metrics["loss"]

        if epoch % settings.checkpoint_every == 0:
            Checkpoint(epoch, is_best=False, path=output_dir)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print_validation_details(model, val_loader, criterion, epoch)
            print(
                "Saving the data for best accuracy of {} obtained at epoch {}".format(
                    best_val_accuracy, epoch
                )
            )
            Checkpoint(
                epoch, is_best=True, path=output_dir
            )  # add checkpoint function here

        if len(test_loader) != 0:
            pbar.log_message(
                "Avg_Train_loss:{:4f}, Val_Loss: {:4f},  Accuracy_val: {:4f}, Test_Loss: {:4f},  Accuracy_test: {:4f}".format(
                    avg_batch_loss, val_loss, val_accuracy, test_loss, test_accuracy
                )
            )

            with open(screen_log, "a") as outfile:
                outfile.write(
                    "{:4f},{:4f},{:4f},{:4f}\n".format(
                        avg_batch_loss, val_loss, val_accuracy, test_accuracy
                    )
                )

        else:
            pbar.log_message(
                "Avg_Train_loss:{:4f}, Val_Loss: {:4f},  Accuracy_val: {:4f}".format(
                    avg_batch_loss, val_loss, val_accuracy
                )
            )
            with open(screen_log, "a") as outfile:
                outfile.write(
                    "{:4f},{:4f},{:4f}\n".format(avg_batch_loss, val_loss, val_accuracy)
                )

        epoch += 1

    trainer.run(train_loader, max_epochs=settings.epochs)


# # Settings

# In[20]:


if os.path.exists("custom_config.yaml"):
    with open("custom_config.yaml", "r") as file:
        custom_dict = yaml.full_load(file)
        settings = Settings(**custom_dict)
else:
    settings = Settings()


# # Loading dataset


poscars = natsorted(glob("{}/*.POSCAR".format(training_data)))

print("about to initialize 'classification_dataset'  as an empty list")

classification_dataset = []

for file in poscars:
    poscar = Poscar.from_file(file)

    dictionary = {
        "structure": poscar.structure,
        "target": np.array(
            [float(lab) for lab in poscar.comment.split(",")], dtype="int"
        ),
    }

    classification_dataset.append(dictionary)

# In[22]:


# -----------data loading-----------------------

graphs = CrystalGraphDataset(
    classification_dataset,
    neighbors=settings.neighbors,
    rcut=settings.rcut,
    delta=settings.search_delta,
)


# # Model

# In[23]:

# start random number seed
random_seed = 430
torch.manual_seed(random_seed)
np.random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# end random number seed

if current_model == "GANN":
    from model import CEGAN
    net = CEGAN(
        settings.gbf_bond,
        settings.gbf_angle,
        n_conv_edge=settings.n_conv_edge,
        h_fea_edge=settings.h_fea_edge,
        h_fea_angle=settings.h_fea_angle,
        n_classification=settings.n_classification,
        pooling=settings.pooling,
        embedding=settings.embedding,
    )
elif current_model == "GIN":
    from model import GIN
    net = GIN(
        settings.gbf_bond,
        settings.gbf_angle,
        n_classification=settings.n_classification,
        neigh=settings.neighbors,
        pooling=settings.pooling,
        embedding=settings.embedding,
    )
elif current_model == "SAGE":
    from model import mySAGE
    net = mySAGE(
        settings.gbf_bond,
        settings.gbf_angle,
        n_classification=settings.n_classification,
        neigh=settings.neighbors,
        pool=settings.pooling,
        embedding=settings.embedding,
    )
elif current_model == "RGCN":
    from model import myRGCN
    net = myRGCN(
        settings.gbf_bond,
        settings.gbf_angle,
        n_classification=settings.n_classification,
        neigh=settings.neighbors,
        pool=settings.pooling,
        embedding=settings.embedding,
    )

# Print the model's parameters
for name, param in net.named_parameters():
    print(name)
# important note: I may need to change from "model" to "net"

print("next line print net")
print(net)

train(graphs, settings, net, output_dir=checkpoint_dir, screen_log=logfile)


