import csv
import logging
import os

import torch
import torch.nn.functional as F
from torch.nn import BCELoss, L1Loss
from torch.utils.data import DataLoader, sampler

from architectures.attribute import GenderClassifier
from architectures.filter import FilterEstimator, IResNetFilter
from architectures.iresnet import iresnet50, iresnet100
from config.config import config as cfg
from utils.data_utils import (
    create_directory,
    make_weights_for_balanced_classes,
)
from utils.dataset import (
    AgeDBDataset,
    ColorFeretDataset,
    LFWDataset,
)
from utils.utils_callbacks import (
    CallBackLogging,
    CallBackModelCheckpoint,
)
from utils.utils_logging import AverageMeter, init_logging


def train():
    torch.cuda.empty_cache()
    device = torch.device("cuda:0")
    csvdir = os.path.join(
        cfg.log_dir, cfg.pretrained, cfg.experiment, cfg.data, cfg.checkpoint_dir
    )
    csvpath = os.path.join(csvdir, "config.csv")

    # Create output and log folder if does not exist
    create_directory(cfg.output)
    create_directory(csvdir)

    # Saving hyperparameters in csv
    header = [
        "output",
        "TrainData",
        "lr-filter",
        "lr-clf",
        "epochs",
        "alpha",
        "beta",
        "batchSize",
        "index",
        "init_zero"
    ]
    file_exists = os.path.isfile(csvpath)
    with open(csvpath, "a", encoding="UTF8") as f:
        cwriter = csv.writer(f)
        if not file_exists:
            cwriter.writerow(header)
        info = [
            cfg.output,
            cfg.data,
            cfg.lr_filter,
            cfg.lr_clf,
            cfg.num_epoch,
            cfg.alpha,
            cfg.beta,
            cfg.batch_size,
            cfg.index,
            cfg.init_zero
        ]
        cwriter.writerow(info)

    # intiate logger to write the output of log file
    log_root = logging.getLogger()
    init_logging(log_root, csvdir)

    # create instance of dataset
    if cfg.data == "ColorFeret":
        trainset = ColorFeretDataset(
            root_dir=os.path.join(cfg.data_dir, "ColorFeret", "ColorFeret_aligned"),
            attribute=os.path.join(
                cfg.data_dir, "colorferet/dvd1/data/ground_truths/xml/subjects.xml"
            ),
        )

    elif cfg.data == "AgeDB":
        trainset = AgeDBDataset(root_dir=os.path.join(cfg.data_dir, "age_db_mtcnn"))
    elif cfg.data == "LFW":
        trainset = LFWDataset(
            root_dir=os.path.join(cfg.data_dir, "lfw_aligned"),
            attribute=os.path.join(cfg.data_dir, "LFW_gender"),
        )
    else:
        exit()

    # Creating trainloader with balanced batches
    weights = make_weights_for_balanced_classes(
        trainset.imgidx, trainset.labels, nclasses=2
    )
    weights = torch.DoubleTensor(weights)
    trainsampler = sampler.WeightedRandomSampler(weights, len(weights))
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=cfg.batch_size,
        sampler=trainsampler,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )

    # load face recognition model
    if cfg.network == "iresnet100":
        backbone = iresnet100(num_features=cfg.embedding_size, use_se=cfg.SE).to(device)
    elif cfg.network == "iresnet50":
        backbone = iresnet50(num_features=cfg.embedding_size, use_se=cfg.SE).to(device)

    try:
        weight = torch.load(cfg.output_ori, map_location=device)
        backbone.load_state_dict(weight)
        print("backbone loaded !")
    except RuntimeError:
        for key in list(weight.keys()):
            weight[key.replace("module.", "")] = weight.pop(key)
        backbone.load_state_dict(weight)

    logging.info("backbone weights loaded successfully!")
    backbone.eval()

    # load gender classifier
    classifier = GenderClassifier(
        input_size=cfg.embedding_size, hidden_size=cfg.embedding_size
    ).to(device)
    weight_g = torch.load(cfg.output_g, map_location=device)
    classifier.load_state_dict(weight_g)
    classifier.eval()
    logging.info("gender classifier loaded !")


    # warm up pass
    _ = backbone(torch.rand(1, 3, 112, 112).to(device), cache_feats=True)
    _readout_feats = backbone.cachefeatures

    """ load pre-calculated stats for the intermediate representation"""
    state = torch.load(
        os.path.join(cfg.estim_dir, "4th_layer.pth"), map_location=device
    )
    n_samples = state["_n_samples"].float()
    std = torch.sqrt(state["s"] / (n_samples - 1)).to(device)
    neuron_nonzero = state["_neuron_nonzero"].float()
    active_neurons = (neuron_nonzero / n_samples) > 0.01
    param_dict = [state["m"].to(device), std, active_neurons]
    logging.info("intermediate representations mean and std loaded successfully!")


    iib = FilterEstimator(_readout_feats[cfg.index].shape[-3], device, param_dict, cfg.init_zero)
    network = IResNetFilter(backbone, iib)

    gcriterion = BCELoss()
    pcriterion = L1Loss()
    opt_iib = torch.optim.Adam(params=[{"params": network.iib.parameters()}], lr=cfg.lr_filter)
    opt_clf = torch.optim.Adam(
        params=[{"params": classifier.parameters()}], lr=cfg.lr_clf
    )

    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size * cfg.num_epoch)

    logging.info("Total Step is: %d" % total_step)

    # verification, logging and checkpoint
    callback_logging = CallBackLogging(cfg.log_freq, total_step, cfg.batch_size)
    callback_checkpoint = CallBackModelCheckpoint(cfg.output)

    loss_r_meter = AverageMeter()
    loss_p_meter = AverageMeter()

    global_step = cfg.global_step
    classifier.eval()
    network.train()
    for p in network.iib.parameters():
        p.requires_grad = True
    for p in network.backbone.parameters():
        p.requires_grad = False
    for p in classifier.parameters():
        p.requires_grad = False


    logging.info("Number of iterations {}".format(len(train_loader)))

    for epoch in range(start_epoch, cfg.num_epoch):
        for i, (img, gender) in enumerate(train_loader):
            global_step += 1
            opt_iib.zero_grad()

            img = img.to(device)
            gender = gender.to(device)
            restrict_emb, lambda_= network(img)
            restrict_emb = F.normalize(restrict_emb)

            gender_probs = classifier(restrict_emb).squeeze()
            mean_lambdas = torch.mean(lambda_)

            # utility loss
            utility_loss = 1.0 - mean_lambdas
            # gender classifier loss
            clf_loss = gcriterion(gender_probs, gender.to(torch.float32))
            # privacy loss
            prv_loss = pcriterion(clf_loss, torch.tensor(cfg.n_classes).log().to(device))

            total_loss = cfg.alpha * utility_loss + cfg.beta * prv_loss
            total_loss.backward()
            opt_iib.step()

            loss_r_meter.update(utility_loss.item(), 1)
            loss_p_meter.update(prv_loss.item(), 1)

            callback_logging(global_step, epoch, loss_r_meter, loss_p_meter)

            # updating the gender classifier
            if ((i + 1) % cfg.freq_clf == 0) and (cfg.freq_clf != 0):
                classifier.train()
                network.eval()
                for p in classifier.parameters():
                    p.requires_grad = True
                for p in network.iib.parameters():
                    p.requires_grad = False

                for _ in range(cfg.iter_clf):
                    opt_clf.zero_grad()
                    restrict_emb, lambda_ = network(img)
                    restrict_emb = F.normalize(restrict_emb)

                    gender_probs = classifier(restrict_emb).squeeze()
                    clf_loss = gcriterion(gender_probs, gender.to(torch.float32))
                    clf_loss.backward()
                    opt_clf.step()
                classifier.eval()
                network.train()
                for p in classifier.parameters():
                    p.requires_grad = False
                for p in network.iib.parameters():
                    p.requires_grad = True
        # save model checkpoint
        callback_checkpoint(global_step, epoch, network)


if __name__ == "__main__":
    train()
