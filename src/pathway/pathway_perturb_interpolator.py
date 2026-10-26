import torch
import torch.nn as nn


class PathwayPerturbInterpolator(nn.Module):


    def __init__(
        self,
        num_bins: int = 5,
        mask_token_id: int = 5,
        mode: str = "off",
        strategy: str = "linear",
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
        mix_ratio: float = 0.5,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.mask_token_id = mask_token_id
        self.mode = self._normalize_mode(mode)
        self.strategy = strategy
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.mix_ratio = mix_ratio

    @staticmethod
    def _normalize_mode(mode) -> str:

        if isinstance(mode, bool):
            return "on" if mode else "off"
        if mode is None:
            return "off"

        mode_str = str(mode).strip().lower()
        if mode_str in {"off", "on", "mixed"}:
            return mode_str
        if mode_str in {"true", "1", "yes", "y"}:
            return "on"
        if mode_str in {"false", "0", "no", "n"}:
            return "off"
        return mode_str

    def _sanitize_scores(self, pathway_scores: torch.Tensor) -> torch.Tensor:
        scores = torch.nan_to_num(pathway_scores, nan=0.0, posinf=self.clamp_max, neginf=self.clamp_min)
        return scores.clamp(self.clamp_min, self.clamp_max)

    def _scores_to_bins(self, pathway_scores: torch.Tensor) -> torch.Tensor:
        scores = self._sanitize_scores(pathway_scores)
        scaled = scores * (self.num_bins - 1)
        return torch.round(scaled).long().clamp(0, self.num_bins - 1)

    def _lookup_bins(
        self,
        embedding_table: torch.Tensor,
        pathway_bins: torch.Tensor,
        pathway_visibility_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        bins = pathway_bins.long().clamp(0, self.mask_token_id)
        if pathway_visibility_mask is not None:
            bins = bins.clone()
            bins[~pathway_visibility_mask] = self.mask_token_id
        return embedding_table[bins]

    def _interpolate_embeddings(
        self,
        embedding_table: torch.Tensor,
        pathway_scores: torch.Tensor,
        pathway_visibility_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        emb_dtype = embedding_table.dtype
        scores = self._sanitize_scores(pathway_scores)
        scaled = scores * (self.num_bins - 1)

        left = torch.floor(scaled).long().clamp(0, self.num_bins - 1)
        right = (left + 1).clamp(0, self.num_bins - 1)

        t = (scaled - left.to(dtype=scaled.dtype)).unsqueeze(-1).to(dtype=emb_dtype)

        left_emb = embedding_table[left]
        right_emb = embedding_table[right]

        if self.strategy == "smoothstep":
            t = t * t * (3.0 - 2.0 * t)

        out = torch.lerp(left_emb, right_emb, t)

        if pathway_visibility_mask is not None:
            mask_emb = embedding_table[self.mask_token_id].view(1, 1, -1)
            out = torch.where(pathway_visibility_mask.unsqueeze(-1), out, mask_emb)

        return out

    def forward(
        self,
        embedding_table: torch.Tensor,
        pathway_bins: torch.Tensor = None,
        pathway_scores: torch.Tensor = None,
        pathway_visibility_mask: torch.Tensor = None,
        force_mode: str = None,
    ) -> torch.Tensor:
        mode = self._normalize_mode(force_mode if force_mode is not None else self.mode)

        if mode == "off":
            if pathway_bins is None:
                if pathway_scores is None:
                    raise ValueError("Need pathway_bins or pathway_scores when mode='off'.")
                pathway_bins = self._scores_to_bins(pathway_scores)
            return self._lookup_bins(embedding_table, pathway_bins, pathway_visibility_mask)

        if mode == "on":
            if pathway_scores is None:
                if pathway_bins is None:
                    raise ValueError("Need pathway_scores or pathway_bins when mode='on'.")
                return self._lookup_bins(embedding_table, pathway_bins, pathway_visibility_mask)
            return self._interpolate_embeddings(embedding_table, pathway_scores, pathway_visibility_mask)

        if mode == "mixed":
            use_interp = False
            if pathway_scores is not None:
                if self.training:
                    use_interp = bool(torch.rand(1).item() < self.mix_ratio)
                else:
                    use_interp = True
            if use_interp:
                return self._interpolate_embeddings(embedding_table, pathway_scores, pathway_visibility_mask)
            if pathway_bins is None:
                if pathway_scores is None:
                    raise ValueError("Need pathway_bins or pathway_scores when mode='mixed'.")
                pathway_bins = self._scores_to_bins(pathway_scores)
            return self._lookup_bins(embedding_table, pathway_bins, pathway_visibility_mask)

        raise ValueError(f"Unsupported interpolation mode: {mode}")
