class AdditionalTermLayer(nn.Module):
    """
    mode:
      'full'            â†’ Full LDAL
      'no_entropy'      â†’ gamma_i = S_i / (1 + max(S))    [H_i removed]
      'no_regularizer'  â†’ r_i = 0 always
      'no_semantic'     â†’ gamma_i = 1 / (1 + H_i)         [S_i removed]
    """
    def __init__(self, target_class_index, num_classes, mode='full'):
        super(AdditionalTermLayer, self).__init__()
        self.target_class_index = target_class_index
        self.num_classes = num_classes
        self.mode = mode
        self.previous_epoch_class_predictions = None
        self.current_epoch_class_predictions = None
        self.current_epoch = -1
        self.norm_sum = None
        self.norm_count = None
        self.entropy_sum = None
        self.entropy_count = None
        self._target_mask = None
        self._device = None

    def _ensure_target_mask(self, device):
        if self._target_mask is None or self._device != device:
            mask = torch.zeros(self.num_classes, dtype=torch.bool, device=device)
            for idx in self.target_class_index:
                mask[idx] = True
            self._target_mask = mask
            self._device = device

    def forward(self, inputs, true_labels, epoch):
        inputs = torch.nan_to_num(inputs)
        device = inputs.device
        B, C = inputs.shape

        if self.current_epoch != epoch:
            if self.current_epoch_class_predictions is not None:
                self.previous_epoch_class_predictions = self.current_epoch_class_predictions.clone()
            self.current_epoch_class_predictions = torch.zeros(C, device=device)
            self.norm_sum = torch.zeros(C, device=device)
            self.norm_count = torch.zeros(C, device=device)
            self.entropy_sum = torch.zeros(C, device=device)
            self.entropy_count = torch.zeros(C, device=device)
            self.current_epoch = epoch

        self._ensure_target_mask(device)

        # 1. Soft predictions (differentiable)
        probs = F.softmax(inputs, dim=-1)
        batch_soft_predictions = probs.sum(dim=0) * (C / B)  # normalized: ~1.0 per class
        with torch.no_grad():
            hard_preds = inputs.argmax(dim=-1)
            hard_counts = torch.zeros(C, device=device)
            hard_counts.scatter_add_(0, hard_preds, torch.ones(B, device=device))
            self.current_epoch_class_predictions += hard_counts

        # 3. Norm statistics
        with torch.no_grad():
            sample_norms = inputs.detach().norm(dim=-1)
            ones_B = torch.ones(B, device=device)
            self.norm_sum.scatter_add_(0, true_labels, sample_norms)
            self.norm_count.scatter_add_(0, true_labels, ones_B)

        # 4. Semantic scales
        with torch.no_grad():
            safe_norm_count = self.norm_count.clamp(min=1)
            avg_magnitudes = self.norm_sum / safe_norm_count
            semantic_scales = avg_magnitudes ** 2
            semantic_scales[self.norm_count == 0] = 0.0
            max_semantic_scale = semantic_scales.max() + 1e-6

        # 5. Class entropies
        with torch.no_grad():
            probs_det = probs.detach()
            sample_entropies = -torch.sum(probs_det * torch.log(probs_det + 1e-6), dim=-1)
            self.entropy_sum.scatter_add_(0, true_labels, sample_entropies)
            self.entropy_count.scatter_add_(0, true_labels, ones_B)
            safe_ent_count = self.entropy_count.clamp(min=1)
            class_entropies = self.entropy_sum / safe_ent_count
            class_entropies[self.entropy_count == 0] = 0.0

        # 6. Dynamic gammas â€” MODE-DEPENDENT
        with torch.no_grad():
            if self.mode == 'no_entropy':
                dynamic_gammas = semantic_scales / (1.0 + max_semantic_scale)
                dynamic_gammas = torch.clamp(dynamic_gammas, max=5.0)
            elif self.mode == 'no_semantic':
                dynamic_gammas = 1.0 / (1.0 + class_entropies)
                dynamic_gammas = torch.clamp(dynamic_gammas, max=5.0)
            else:  # 'full' or 'no_regularizer'
                dynamic_gammas = semantic_scales / (1.0 + max_semantic_scale * class_entropies)
            dynamic_gammas = torch.clamp(dynamic_gammas, max=5.0)

        # 7. Reinforcement terms â€” MODE-DEPENDENT
        with torch.no_grad():
            reinforcement_terms = torch.zeros(C, device=device)
            if self.mode != 'no_regularizer' and self.previous_epoch_class_predictions is not None:
                cur = self.current_epoch_class_predictions
                prev = self.previous_epoch_class_predictions
                mask = self._target_mask
                reinforcement_terms[(cur > prev) & mask] = -2.0
                reinforcement_terms[(cur < prev) & mask] = 2.0

        # 8. LDAL loss
        numerators = (dynamic_gammas * batch_soft_predictions + reinforcement_terms) ** 2
        detached = inputs.detach()
        total_sq = (detached ** 2).sum()
        col_sums = detached.sum(dim=0)
        denominators = (total_sq - 2.0 * col_sums + B) / (B * C) + 1.0
        additional_term = (numerators / denominators).sum() / C

        return additional_term


class CustomLossWithLDAL(nn.Module):
    def __init__(self, target_class_index, num_classes, mode='full'):
        super(CustomLossWithLDAL, self).__init__()
        self.additional_term_layer = AdditionalTermLayer(target_class_index, num_classes, mode=mode)

    def forward(self, y_true, y_pred, epoch):
        cross_entropy_loss = F.cross_entropy(y_pred, y_true.squeeze().long())
        additional_term = self.additional_term_layer(y_pred, y_true, epoch)
        total_loss = cross_entropy_loss + additional_term
        return total_loss


class CEOnlyLoss(nn.Module):
    """Plain cross-entropy for the baseline variant."""
    def forward(self, y_true, y_pred, epoch):
        return F.cross_entropy(y_pred, y_true.squeeze().long())
