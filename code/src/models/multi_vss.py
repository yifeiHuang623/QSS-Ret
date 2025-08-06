import os.path as osp
import torch
from typing import Any
from src.models.model import ModelForSemiStructQA
from src.models.vss import VSS
from src.tools.process_text import chunk_text
from sentence_transformers import SentenceTransformer


class MultiVSS(ModelForSemiStructQA):
    
    def __init__(self, 
                 kb,
                 query_emb_dir,
                 candidates_emb_dir,
                 chunk_emb_dir,
                 emb_model='text-embedding-ada-002',
                 aggregate='top3_avg',
                 max_k=100,
                 chunk_size=256):
        '''
        Multivector Vector Similarity Search
        Args:
            kb (src.benchmarks.semistruct.SemiStruct): kb
            query_emb_dir (str): directory to query embeddings
            candidates_emb_dir (str): directory to candidate embeddings
            chunk_emb_dir (str): directory to chunk embeddings
        '''
        
        super().__init__(kb)
        self.kb = kb
        self.aggregate = aggregate # 'max', 'avg', 'top{k}_avg'

        self.max_k = max_k
        self.chunk_size = chunk_size
        self.emb_model = emb_model
        self.query_emb_dir = query_emb_dir
        self.chunk_emb_dir = chunk_emb_dir
        self.candidates_emb_dir = candidates_emb_dir
        self.parent_vss = VSS(kb, query_emb_dir, candidates_emb_dir, emb_model=emb_model)

        self.emb_model = SentenceTransformer("../../llm_model/gte-large-en-v1.5", trust_remote_code=True)

    def forward(self, 
                query,
                query_id,
                truth = [],
                **kwargs: Any):
        
        query_emb = self._get_query_emb(query, query_id)

        initial_score_dict, _, _ = self.parent_vss(query, query_id)
        node_ids = list(initial_score_dict.keys())
        node_scores = list(initial_score_dict.values())

        # get the ids with top k highest scores
        top_k_idx = torch.topk(torch.FloatTensor(node_scores),
                               min(self.max_k, len(node_scores)),
                               dim=-1
                               ).indices.view(-1).tolist()
        top_k_node_ids = [node_ids[i] for i in top_k_idx]

        pred_dict = {}
        for node_id in top_k_node_ids:
        # for node_id in (truth + top_k_node_ids[:10]):
            doc = self.kb.get_doc_info(node_id, add_rel=True, compact=True)
            chunks = chunk_text(doc, chunk_size=self.chunk_size)
            chunk_path = osp.join(self.chunk_emb_dir, f'{node_id}_size={self.chunk_size}.pt')
            if osp.exists(chunk_path):
                chunk_embs = torch.load(chunk_path)
            else:
                chunk_embs = torch.from_numpy(self.emb_model.encode(chunks))
                torch.save(chunk_embs, chunk_path)
            print(f'chunk_embs.shape: {chunk_embs.shape}')

            similarity = torch.matmul(query_emb.cuda(), chunk_embs.cuda().T).cpu().view(-1)

            if self.aggregate == 'max':
                pred_dict[node_id] = torch.max(similarity).item()
            elif self.aggregate == 'avg':
                pred_dict[node_id] = torch.mean(similarity).item()
            elif 'top' in self.aggregate:
                k = int(self.aggregate.split('_')[0][len('top'):])
                pred_dict[node_id] = torch.mean(torch.topk(similarity, k=min(k, len(chunks)), dim=-1).values).item()

        node_ids = list(pred_dict.keys())
        node_scores = list(pred_dict.values())
        top_k_idx = torch.topk(torch.FloatTensor(node_scores),
                               min(self.max_k, len(node_scores)),
                               dim=-1).indices.view(-1).tolist()
        top_k_node_ids = [node_ids[i] for i in top_k_idx]

        return pred_dict, top_k_node_ids