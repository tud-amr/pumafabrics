import torch
import torch.nn as nn
import torch.nn.functional as F
cosine_loss = torch.nn.CosineSimilarity(dim=1)
"""
Source: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
"""


def great_circle_distance(a, b):
    cosine = torch.clamp(cosine_loss(a, b), -0.99999, 0.99999)  # clamp necessary to avoid nans
    distance = torch.acos(cosine)
    return distance


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive + self.margin - distance_negative)**2
        return losses.mean()


class SoftTripletLoss(nn.Module):
    """
    Soft Triplet loss
    Soft triplet loss from "ThirdEye: Triplet Based Iris Recognition without Normalization"
    """

    def __init__(self):
        super(SoftTripletLoss, self).__init__()

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.log(1 + torch.exp(distance_positive - distance_negative))
        return losses.mean()


class TripletAngleLoss(nn.Module):
    """
    Triplet Angle loss
    Triplet loss using angle distance as metric
    """

    def __init__(self, margin):
        super(TripletAngleLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = great_circle_distance(anchor, positive)
        distance_negative = great_circle_distance(anchor, negative)
        losses = F.relu(distance_positive + self.margin - distance_negative)**2
        return losses.mean()


class TripletCosineLoss(nn.Module):
    """
    Triplet Cosine loss
    Triplet loss using the cosine distance as a "metric"
    """

    def __init__(self, margin):
        super(TripletCosineLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        cosine_loss = torch.nn.CosineSimilarity(dim=1)
        distance_positive = (1 - cosine_loss(anchor, positive))
        distance_negative = (1 - cosine_loss(anchor, negative))
        losses = F.relu(distance_positive + self.margin - distance_negative)**2 + 1e-20
        return losses.mean()
