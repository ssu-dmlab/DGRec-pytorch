#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


class MyEvaluator:
    def __init__(self, device):
        self.device = device

    def evaluate(self, model, test_data):
        with torch.no_grad():
            model.eval()
            features = test_data.get_features().view(-1, test_data.in_dim).to(self.device)
            labels = test_data.get_labels().to(self.device)

            predictions = model.predict(features)
            corrects = torch.argmax(predictions, 1) == labels
            accuracy = corrects.float().mean()

        return accuracy.item()
