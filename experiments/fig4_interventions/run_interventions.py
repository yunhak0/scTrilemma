"""Demand-targeted latent interventions on the trained scTrilemma checkpoint.

For each held-out dataset (the stratified 2,500-cell sample from ``experiments/prepare_samples.py``),
the posterior-mean
latent tokens ``T`` (N, M, D) are intervened at inference time and the three demands are
measured on the *same* intervened latent:

* identity / invariance on the pooled embedding ``T'.mean(1)`` (scib-metrics protocol);
* expression fidelity by decoding ``T'`` with the unchanged decoder over the shared gene list.

Interventions (strength alpha in [0, 1]):

* ``donor_centering``  -> T' = T - alpha * (mu_donor - mu_all)          (pushes context invariance)
* ``centroid_shrinkage`` -> T' = (1 - alpha) * T + alpha * c_k            (pushes biological identity;
  k-means on the pooled embedding with K = number of annotated cell types, seed 0)
* ``label_shrinkage``    -> same, but c_k is the annotated cell-type centroid   (oracle identity push; raises NMI/ARI)
* ``latent_refinement``  -> T' = T + alpha * (T* - T), T* = test-time optimisation of T for per-cell
  reconstruction Pearson (or count NLL) of the same cell                         (pushes expression fidelity)

alpha = 0 is the unmodified model and is scored once per dataset.

Run ``experiments/fig4_interventions/run_all.sh`` for the full set of alpha grids used in the paper.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from experiments.common import (
    dense_columns,
    feature_lookup,
    load_sampled_adata,
    log1p_cp10k,
    normalize_gene,
    read_ids,
    score_matrix,
)
from sctrilemma.benchmark.models import ScTrilemmaInference
from sctrilemma.inference import (
    _build_col_to_vocab,
    _build_donor_context,
    _compute_full_cell_library,
    _preprocess_cell_batch,
    _resolve_gene_vocab,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SCTRILEMMA_DATA_ROOT", f"/scratch/{os.environ.get('USER', 'user')}/datasets/cellxgene"))
DEFAULT_CKPT = ROOT / "checkpoints/sctrilemma/final.ckpt"
FIELDS = ["dataset", "intervention", "alpha", "metric", "value", "n_cells", "n_cell_types", "n_donors", "n_genes"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-ids-file", type=Path, default=ROOT / "configs/zsb/full_89_ids.txt")
    p.add_argument("--dataset-ids", nargs="+")
    p.add_argument("--data-dir", type=Path, default=ROOT / "outputs/experiments/samples",
                   help="Sampled datasets written by experiments/prepare_samples.py")
    p.add_argument("--genelist-dir", type=Path, default=ROOT / "experiments/data/genelists")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--gene-vocab", type=Path, default=DATA_ROOT / "20250130/gene_vocab_homo_sapiens_20250130.json")
    p.add_argument("--pseudo-bulk", type=Path, default=None, help="Optional pseudo-bulk dict (not needed for inference)")
    p.add_argument("--tissue-code", type=Path, default=None, help="Optional tissue-code dict (not needed for inference)")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiments/fig4_interventions")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--interventions", nargs="+", default=["donor_centering", "centroid_shrinkage"],
                   help="donor_centering | centroid_shrinkage | label_shrinkage | latent_refinement")
    p.add_argument("--refine-steps", type=int, default=30, help="Adam steps of test-time latent refinement")
    p.add_argument("--refine-lr-scale", type=float, default=0.02, help="Adam lr as a fraction of the token std")
    p.add_argument("--refine-objective", choices=["pearson", "nll"], default="pearson",
                   help="pearson: maximise per-cell Pearson of log1p(CP10K) decoded vs observed; nll: minimise the count NLL")
    p.add_argument("--kmeans-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--min-detection-rate", type=float, default=0.02)
    p.add_argument("--label-key", default="cell_type")
    p.add_argument("--batch-key", default="donor_id")
    return p.parse_args()


# ----------------------------------------------------------------------------- model access
class LatentAccess:
    """Encode to posterior-mean tokens and decode arbitrary tokens with the same checkpoint."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.wrapper = ScTrilemmaInference(
            checkpoint_path=str(args.checkpoint),
            gene_vocab_path=str(args.gene_vocab),
            pseudo_bulk_path=str(args.pseudo_bulk) if args.pseudo_bulk else None,
            tissue_code_path=str(args.tissue_code) if args.tissue_code else None,
        )
        self.wrapper.load_model()
        self.pl_module = self.wrapper.pl_module
        self.pl_module.eval()
        self.model = self.pl_module.model
        self.dev = next(self.model.parameters()).device
        self.gene_vocab = _resolve_gene_vocab(None, self.wrapper._gene_vocab)
        self.crop = int(self.wrapper._encode_crop_size)
        self.batch_size = int(args.batch_size)

    def prepare(self, adata, decode_genes: list[str]) -> None:
        self.adata = adata
        self.col_to_vocab = _build_col_to_vocab(adata, self.gene_vocab)
        self.donor_codes, self.tissue_code_per_cell, self.pb_table = _build_donor_context(
            adata, self.wrapper._tissue_code_dict, self.wrapper._pseudo_bulk_dict, self.model, len(self.gene_vocab)
        )
        var_to_col = {str(n): i for i, n in enumerate(adata.var_names)}
        file_idx = [var_to_col[g] for g in decode_genes if g in var_to_col and var_to_col[g] in self.col_to_vocab]
        if not file_idx:
            raise ValueError("no decode gene present in adata and vocab")
        self.decode_gene_names = [str(adata.var_names[i]) for i in file_idx]
        self.decode_file_idx = file_idx
        self.decode_vocab_t = torch.tensor([self.col_to_vocab[i] for i in file_idx], dtype=torch.long, device=self.dev)
        self.full_library = _compute_full_cell_library(adata)

    def _batch_context(self, start: int, end: int):
        tissue = self.tissue_code_per_cell[start:end].to(self.dev) if self.tissue_code_per_cell is not None else None
        pb = None
        if self.pb_table is not None:
            pb = self.pb_table[torch.from_numpy(self.donor_codes[start:end]).long()].to(self.dev)
        return tissue, pb

    @torch.no_grad()
    def encode_batch(self, start: int, end: int):
        tissue, pb = self._batch_context(start, end)
        idx, val, mask = _preprocess_cell_batch(self.adata.X[start:end], self.col_to_vocab, self.crop, 10000.0, self.dev)
        z, mu, _, _, z_gene = self.model.encode(
            meta_features={}, raw_input=val, mask=mask, gene_indices=idx, pseudo_bulk=pb, tissue_code=tissue
        )
        assert torch.equal(z, mu), "eval-mode reparameterisation must return mu"
        return mu, z_gene, tissue, pb

    @torch.no_grad()
    def encode_all(self) -> torch.Tensor:
        n = self.adata.n_obs
        tokens = None
        for s in range(0, n, self.batch_size):
            e = min(s + self.batch_size, n)
            mu, _, _, _ = self.encode_batch(s, e)
            if tokens is None:
                tokens = torch.empty((n, *mu.shape[1:]), dtype=torch.float32, device=self.dev)
            tokens[s:e] = mu.float()
        return tokens

    @torch.no_grad()
    def decode_tokens(self, tokens_by_condition: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        """Decode several token tensors (same cells) sharing one encoder pass per batch."""
        n = self.adata.n_obs
        g = len(self.decode_gene_names)
        out = {k: np.zeros((n, g), dtype=np.float32) for k in tokens_by_condition}
        for s in range(0, n, self.batch_size):
            e = min(s + self.batch_size, n)
            mu, z_gene, tissue, pb = self.encode_batch(s, e)
            b = e - s
            decode_idx = self.decode_vocab_t.unsqueeze(0).expand(b, -1)
            decode_mask = torch.ones(b, g, dtype=torch.bool, device=self.dev)
            gene_embs = self.model._build_gene_embeddings({}, decode_idx)
            lib = self.full_library[s:e].to(self.dev)
            for k, tok in tokens_by_condition.items():
                z = tok[s:e].to(mu.dtype)
                zinb_mu, _, _, _ = self.model.decode(
                    z, gene_embs, context=pb, gene_indices=decode_idx, padding_mask=decode_mask,
                    library_size=lib, z_gene=z_gene, tissue_code=tissue,
                )
                out[k][s:e] = zinb_mu.float().cpu().numpy()
        return out

    def refine_all(self, tokens: torch.Tensor, steps: int, lr_scale: float, objective: str = "pearson") -> tuple[torch.Tensor, float, float]:
        """Test-time latent refinement: minimise the model's own count NLL over the decode genes w.r.t. z.

        Returns the refined tokens (same shape as ``tokens``) and the mean NLL before/after.
        """
        n = self.adata.n_obs
        g = len(self.decode_gene_names)
        out = torch.empty_like(tokens)
        loss_fn = self.pl_module.recon_loss_fn
        nll_before, nll_after = [], []
        for prm in self.model.parameters():
            prm.requires_grad_(False)
        for s in range(0, n, self.batch_size):
            e = min(s + self.batch_size, n)
            b = e - s
            with torch.no_grad():
                mu, z_gene, tissue, pb = self.encode_batch(s, e)
                decode_idx = self.decode_vocab_t.unsqueeze(0).expand(b, -1)
                decode_mask = torch.ones(b, g, dtype=torch.bool, device=self.dev)
                gene_embs = self.model._build_gene_embeddings({}, decode_idx)
                lib = self.full_library[s:e].to(self.dev)
                target = torch.from_numpy(np.asarray(dense_columns(self.adata.X[s:e], self.decode_file_idx), dtype=np.float32)).to(self.dev)
            z = tokens[s:e].clone().to(mu.dtype).requires_grad_(True)
            opt = torch.optim.Adam([z], lr=lr_scale * float(tokens.std()))
            for step in range(steps + 1):
                with torch.enable_grad():
                    zinb_mu, theta, pi, _ = self.model.decode(
                        z, gene_embs, context=pb, gene_indices=decode_idx, padding_mask=decode_mask,
                        library_size=lib, z_gene=z_gene, tissue_code=tissue,
                    )
                    if objective == "pearson":
                        pred = torch.log1p(zinb_mu.float() / zinb_mu.float().sum(dim=1, keepdim=True).clamp_min(1e-8) * 1e4)
                        obs = torch.log1p(target / target.sum(dim=1, keepdim=True).clamp_min(1e-8) * 1e4)
                        pc = pred - pred.mean(dim=1, keepdim=True)
                        oc = obs - obs.mean(dim=1, keepdim=True)
                        corr = (pc * oc).sum(dim=1) / (pc.norm(dim=1) * oc.norm(dim=1)).clamp_min(1e-8)
                        loss = -corr.mean()
                    elif self.pl_module.gene_likelihood == "nb":
                        loss = loss_fn(zinb_mu.float(), theta.float(), target, decode_mask)
                    else:
                        loss = loss_fn(zinb_mu.float(), theta.float(), pi.float(), target, decode_mask)
                if step == 0:
                    nll_before.append(float(loss))
                if step == steps:
                    nll_after.append(float(loss))
                    break
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            out[s:e] = z.detach().float()
        return out, float(np.mean(nll_before)), float(np.mean(nll_after))

    @torch.no_grad()
    def pooled(self, tokens: torch.Tensor) -> np.ndarray:
        # no_grad: a parametric pooler (attention pooling) would otherwise return a grad-requiring tensor
        return self.model.get_representation(tokens).detach().float().cpu().numpy()


# ----------------------------------------------------------------------------- interventions
def group_means(tokens: torch.Tensor, groups: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    codes, inverse = np.unique(groups, return_inverse=True)
    inv = torch.from_numpy(inverse).to(tokens.device)
    sums = torch.zeros((len(codes), *tokens.shape[1:]), dtype=tokens.dtype, device=tokens.device)
    sums.index_add_(0, inv, tokens)
    counts = torch.bincount(inv, minlength=len(codes)).to(tokens.dtype).view(-1, *([1] * (tokens.dim() - 1)))
    return sums / counts, inverse


def kmeans_labels(embedding: np.ndarray, k: int, seed: int) -> np.ndarray:
    from scib_metrics.utils import KMeans

    return np.asarray(KMeans(n_clusters=k, seed=seed).fit(embedding).labels_)


def donor_centering(tokens: torch.Tensor, donors: np.ndarray, alpha: float) -> torch.Tensor:
    means, inverse = group_means(tokens, donors)
    mu_all = tokens.mean(dim=0, keepdim=True)
    offset = means[torch.from_numpy(inverse).to(tokens.device)] - mu_all
    return tokens - alpha * offset


def centroid_shrinkage(tokens: torch.Tensor, clusters: np.ndarray, alpha: float) -> torch.Tensor:
    means, inverse = group_means(tokens, clusters)
    centroid = means[torch.from_numpy(inverse).to(tokens.device)]
    return (1.0 - alpha) * tokens + alpha * centroid


# ----------------------------------------------------------------------------- metrics
def embedding_metrics(X: np.ndarray, labels: np.ndarray, batches: np.ndarray, seeds: list[int]) -> dict[str, float]:
    from scib_metrics import bras, clisi_knn, ilisi_knn, isolated_labels, silhouette_label
    from scib_metrics.nearest_neighbors import pynndescent

    out: dict[str, float] = {}
    k = len(np.unique(labels))
    nmi, ari = [], []
    for seed in seeds:
        pred = kmeans_labels(X, k, seed)
        nmi.append(normalized_mutual_info_score(labels, pred, average_method="arithmetic"))
        ari.append(adjusted_rand_score(labels, pred))
    out["nmi"] = float(np.mean(nmi))
    out["ari"] = float(np.mean(ari))
    out["silhouette_label"] = float(silhouette_label(X, labels))
    neighbors = pynndescent(X, n_neighbors=90)
    out["clisi_knn"] = float(clisi_knn(neighbors, labels))
    if len(np.unique(batches)) > 1:
        out["isolated_labels"] = float(isolated_labels(X, labels, batches))
        out["ilisi_knn"] = float(ilisi_knn(neighbors, batches))
        out["bras"] = float(bras(X, labels, batches))
    return out


def fidelity_reference(adata, decode_gene_names: list[str], min_detection_rate: float):
    """Raw log1p(CP10K) matrix over decoded genes passing the detection filter, plus recon columns."""
    raw_lookup = feature_lookup(adata)
    recon_map: dict[str, int] = {}
    for i, g in enumerate(decode_gene_names):
        recon_map.setdefault(normalize_gene(g), i)
    ordered = [g for g, _ in sorted(raw_lookup.items(), key=lambda kv: kv[1]) if g in recon_map]
    raw_counts = np.asarray(dense_columns(adata.X, [raw_lookup[g] for g in ordered]))
    if min_detection_rate > 0:
        keep = (raw_counts > 0).mean(axis=0) >= min_detection_rate
        ordered = [g for g, k in zip(ordered, keep) if k]
        raw_counts = raw_counts[:, keep]
    return log1p_cp10k(raw_counts), [recon_map[g] for g in ordered]


# ----------------------------------------------------------------------------- main
def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids = args.dataset_ids or read_ids(args.dataset_ids_file)
    out_path = args.output_dir / "interventions_long.csv"
    done: set[str] = set()
    if out_path.exists():
        with out_path.open() as f:
            done = {r["dataset"] for r in csv.DictReader(f)}
    handle = out_path.open("a", newline="")
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    if not done:
        writer.writeheader()
    (args.output_dir / "run_config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    access = LatentAccess(args)
    device = access.dev
    for i, ds in enumerate(ids, 1):
        if ds in done:
            continue
        t0 = time.time()
        adata = load_sampled_adata(args.data_dir / f"{ds}.h5ad", ds)
        genelist = [g.strip() for g in (args.genelist_dir / f"{ds}.txt").read_text().splitlines() if g.strip()]
        access.prepare(adata, genelist)
        labels = adata.obs[args.label_key].astype(str).to_numpy()
        donors = adata.obs[args.batch_key].astype(str).to_numpy()
        n_types, n_donors = len(np.unique(labels)), len(np.unique(donors))

        tokens = access.encode_all()
        base_embedding = access.pooled(tokens)
        clusters = kmeans_labels(base_embedding, n_types, 0)

        conditions: dict[tuple[str, float], torch.Tensor] = {("none", 0.0): tokens}
        refined = None
        if "latent_refinement" in args.interventions:
            refined, nll0, nll1 = access.refine_all(tokens, args.refine_steps, args.refine_lr_scale, args.refine_objective)
            print(f"    refinement objective ({args.refine_objective}) {nll0:.4f} -> {nll1:.4f}", flush=True)
        for name in args.interventions:
            if name == "donor_centering" and n_donors < 2:
                continue
            for a in args.alphas:
                if name == "donor_centering":
                    conditions[(name, a)] = donor_centering(tokens, donors, a)
                elif name == "centroid_shrinkage":
                    conditions[(name, a)] = centroid_shrinkage(tokens, clusters, a)
                elif name == "label_shrinkage":
                    conditions[(name, a)] = centroid_shrinkage(tokens, labels, a)
                elif name == "latent_refinement":
                    conditions[(name, a)] = tokens + a * (refined - tokens)
                else:
                    raise ValueError(name)

        raw, recon_cols = fidelity_reference(adata, access.decode_gene_names, args.min_detection_rate)
        recons = access.decode_tokens({f"{k[0]}|{k[1]}": v for k, v in conditions.items()})

        rows = []
        for (name, a), tok in conditions.items():
            metrics = embedding_metrics(access.pooled(tok), labels, donors, args.kmeans_seeds)
            recon = log1p_cp10k(recons[f"{name}|{a}"][:, recon_cols])
            metrics.update(score_matrix(raw, recon, device=device))
            for m, v in metrics.items():
                rows.append({"dataset": ds, "intervention": name, "alpha": a, "metric": m, "value": v,
                             "n_cells": adata.n_obs, "n_cell_types": n_types, "n_donors": n_donors, "n_genes": raw.shape[1]})
        writer.writerows(rows)
        handle.flush()
        del tokens, conditions, recons
        torch.cuda.empty_cache()
        print(f"[{i}/{len(ids)}] {ds[:8]} cells={adata.n_obs} types={n_types} donors={n_donors} genes={raw.shape[1]} "
              f"conditions={len(rows) // max(1, len(metrics))} {time.time() - t0:.0f}s", flush=True)
    handle.close()
    print("Done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
