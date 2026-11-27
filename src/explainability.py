import io
import base64
import numpy as np
import torch
from PIL import Image
from skimage.segmentation import slic, mark_boundaries
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import Ridge
import matplotlib.cm as cm
from src.inference import LIMIAR_MALIGNO

class SuperpixelGraphExplainer:
    def __init__(
        self,
        model: torch.nn.Module,
        n_segments: int = 70,
        compactness: float = 10.0,
        num_samples: int = 150,
        kernel_width: float = 0.25,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval() #
        self.n_segments = n_segments
        self.compactness = compactness
        self.num_samples = num_samples
        self.kernel_width = kernel_width
        self.device = device
        hparams = getattr(model, "hparams", None)
        self.input_height = getattr(hparams, "input_height", 224)
        self.input_width = getattr(hparams, "input_width", 224)
    def build_superpixel_graph(self, img_np: np.ndarray):
        """
        Segmenta a imagem (224x224x3, float [0,1]) via SLIC e constrói:
        - segments: mapa pixel -> ID do nó
        - node_features: cor média RGB e centroide (x, y) de cada nó
        - adj_matrix: matriz de adjacência ponderada por similaridade visual
        """
        segments = slic(
            img_np,
            n_segments=self.n_segments,
            compactness=self.compactness,
            start_label=0,
        )
        num_nodes = int(segments.max() + 1)
        h, w, _ = img_np.shape

        # Atributos dos nós: [R_mean, G_mean, B_mean, y_norm, x_norm]
        node_features = np.zeros((num_nodes, 5), dtype=np.float32)
        yy, xx = np.mgrid[0:h, 0:w]

        for node_id in range(num_nodes):
            mask = segments == node_id
            if np.any(mask):
                node_features[node_id, :3] = img_np[mask].mean(axis=0)
                node_features[node_id, 3] = yy[mask].mean() / h
                node_features[node_id, 4] = xx[mask].mean() / w

        # Arestas por adjacência espacial (vizinhos 4-conectados)
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        # Vizinhos horizontais e verticais
        diff_h = segments[:, :-1] != segments[:, 1:]
        pairs_h = np.stack([segments[:, :-1][diff_h], segments[:, 1:][diff_h]], axis=1)

        diff_v = segments[:-1, :] != segments[1:, :]
        pairs_v = np.stack([segments[:-1, :][diff_v], segments[1:, :][diff_v]], axis=1)

        all_pairs = np.vstack([pairs_h, pairs_v])
        for u, v in all_pairs:
            if u != v and adj[u, v] == 0:
                # Similaridade de cor (RBF kernel sobre RGB médio)
                color_dist = np.linalg.norm(node_features[u, :3] - node_features[v, :3])
                weight = np.exp(-(color_dist ** 2) / 0.1)
                adj[u, v] = weight
                adj[v, u] = weight

        return segments, node_features, adj

    def _sample_graph_perturbations(self, adj: np.ndarray, rng: np.random.Generator):
        """
        Gera máscaras binárias Z (num_samples x num_nodes) respeitando a estrutura
        do grafo (propaga ativação/desativação para vizinhos fortemente conectados).
        """
        num_nodes = adj.shape[0]
        Z = np.ones((self.num_samples, num_nodes), dtype=np.float32)

        # Primeira amostra é sempre a imagem original intacta
        for i in range(1, self.num_samples):
            # Sorteia nós sementes para desligar (0)
            drop_prob = rng.uniform(0.15, 0.55)
            seeds = rng.random(num_nodes) < drop_prob
            mask_off = seeds.copy()

            # Propaga 1 salto no grafo proporcional ao peso da aresta
            for node in np.where(seeds)[0]:
                neighbors = np.where(adj[node] > 0)[0]
                for nb in neighbors:
                    if rng.random() < 0.5 * adj[node, nb]:
                        mask_off[nb] = True

            Z[i, mask_off] = 0.0

        return Z

    @torch.no_grad()
    def _predict_batch(self, imgs_np: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Executa inferência em lotes na CNN."""
        preds = []
        # [N, H, W, C] -> [N, C, H, W]
        tensors = torch.from_numpy(imgs_np.transpose(0, 3, 1, 2)).float()
        for start in range(0, len(tensors), batch_size):
            batch = tensors[start : start + batch_size].to(self.device)
            out = self.model(batch).squeeze(-1)
            preds.append(out.cpu().numpy())
        return np.concatenate(preds, axis=0)

    def explain(self, pil_image: Image.Image, seed: int = 42):
        """
        Gera explicação completa para uma imagem PIL:
        - prob_original: predição da CNN
        - weights: importância de cada superpixel
        - heatmap_overlay_b64: imagem PNG em base64 pronta para a API Flask
        - fidelity_drop: queda de probabilidade ao remover os top-5 superpixels
        """
        rng = np.random.default_rng(seed)
        img_resized = pil_image.convert("RGB").resize((self.input_width, self.input_height))
        img_np = np.asarray(img_resized, dtype=np.float32) / 255.0

        segments, node_features, adj = self.build_superpixel_graph(img_np)
        num_nodes = adj.shape[0]

        Z = self._sample_graph_perturbations(adj, rng)
        baseline_color = img_np.mean(axis=(0, 1))  # Cor média para oclusão

        # Constrói imagens perturbadas
        perturbed_imgs = np.empty((self.num_samples, self.input_height, self.input_width, 3))

        for i in range(self.num_samples):
            active_mask = Z[i][segments] > 0.5  # [224, 224]
            img_p = np.where(active_mask[..., None], img_np, baseline_color)
            perturbed_imgs[i] = img_p

        probs = self._predict_batch(perturbed_imgs)
        prob_original = float(probs[0])

        # Distância cosseno / L2 entre Z e o vetor all-ones (imagem original)
        distances = np.linalg.norm(1.0 - Z, axis=1) / np.sqrt(num_nodes)
        sample_weights = np.exp(-(distances ** 2) / (self.kernel_width ** 2))

        # Modelo surrogate local (Ridge esparso/regularizado)
        surrogate = Ridge(alpha=1.0)
        surrogate.fit(Z, probs, sample_weight=sample_weights)
        node_weights = surrogate.coef_

        # Projeta pesos nos pixels e normaliza para visualização
        pixel_map = node_weights[segments]
        pixel_map = gaussian_filter(pixel_map, sigma=3.0)

        
        # Foco nas regiões que contribuem positivamente para a classe predita
        target_map = pixel_map if prob_original > LIMIAR_MALIGNO else -pixel_map
        pos_map = np.maximum(target_map, 0)
        if pos_map.max() > 0:
            pos_map = pos_map / pos_map.max()

        # Cria overlay com colormap 'jet' + contornos SLIC sutis
        heatmap_rgb = cm.jet(pos_map)[..., :3]
        overlay = 0.60 * img_np + 0.40 * heatmap_rgb
        overlay = mark_boundaries(overlay, segments, color=(1, 1, 1), mode="subpixel")
        overlay_uint8 = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

        # Converte para PNG base64
        buf = io.BytesIO()
        Image.fromarray(overlay_uint8).save(buf, format="PNG")
        overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Teste rápido de fidelidade (mascarando top-5 superpixels)
        top_k = min(5, num_nodes)
        order = np.argsort(-node_weights if prob_original > LIMIAR_MALIGNO else node_weights)
        top_nodes = order[:top_k]
        mask_fidelity = np.isin(segments, top_nodes)
        img_masked = np.where(mask_fidelity[..., None], baseline_color, img_np)
        prob_masked = float(self._predict_batch(img_masked[None, ...])[0])
        fidelity_drop = abs(prob_original - prob_masked)

        return {
            "probabilidade": prob_original,
            "num_superpixels": num_nodes,
            "num_arestas": int(np.sum(adj > 0) // 2),
            "node_weights": node_weights.tolist(),
            "fidelity_drop_top5": fidelity_drop,
            "overlay_image": overlay_uint8,
            "overlay_base64": overlay_b64,
        }