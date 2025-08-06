import torch
from typing import Any

from src.models.vss import VSS
from src.models.multi_vss import MultiVSS
from src.models.model import ModelForSemiStructQA
from src.tools.api import get_llm_output
import re
from tqdm import tqdm


def find_floating_number(text):
    pattern = r'0\.\d+|1\.0|0|1'
    matches = re.findall(pattern, text)
    return [round(float(match), 4) for match in matches if float(match) <= 1.1]


class LLMReranker(ModelForSemiStructQA):
    
    def __init__(self,
                 kb,
                 emb_model,
                 llm_model_pt,
                 tokenizer, 
                 llm_model,
                 query_emb_dir, 
                 candidates_emb_dir,
                 chunk_emb_dir,
                 sim_weight=0.01,
                 max_cnt=3,
                 max_k=100,
                 dataset='amazon'
                 ):
        '''
        Answer the query by GPT model.
        Args:
            kb (src.benchmarks.semistruct.SemiStruct): kb
            llm_model (str): model name
            query_emb_dir (str): directory to query embeddings
            candidates_emb_dir (str): directory to candidate embeddings  
        '''
        
        super(LLMReranker, self).__init__(kb)
        self.max_k = max_k
        self.kb = kb
        self.emb_model = emb_model
        self.llm_model_name = llm_model
        self.sim_weight = sim_weight
        self.max_cnt = max_cnt
        self.llm_model_pt = llm_model_pt
        self.tokenizer = tokenizer

        self.query_emb_dir = query_emb_dir
        self.candidates_emb_dir = candidates_emb_dir
        
        if dataset == "amazon":
            self.parent_vss = MultiVSS(kb, query_emb_dir, candidates_emb_dir, emb_model=emb_model, chunk_emb_dir=chunk_emb_dir, aggregate='max', max_k=50)
        else:
            self.parent_vss = VSS(kb, query_emb_dir, candidates_emb_dir, emb_model=emb_model)


    def forward(self, 
                query,
                query_id=None,
                mode='test',
                truth=[],
                **kwargs: Any):

        initial_score_dict = self.parent_vss(query, query_id)[0]
        node_ids = list(initial_score_dict.keys())
        node_scores = list(initial_score_dict.values())

        # get the ids with top k highest scores
        top_k_idx = torch.topk(torch.FloatTensor(node_scores),
                               min(self.max_k, len(node_scores)),
                               dim=-1).indices.view(-1).tolist()
        top_k_node_ids = [node_ids[i] for i in top_k_idx]
        cand_len = len(top_k_node_ids)

        pred_dict = {}
        print(f'top_k_node_ids count is {cand_len}')

        for idx, node_id in tqdm(enumerate(top_k_node_ids)):
            node_type = self.kb.get_node_type_by_id(node_id)

            prompt = (
                        f"You are a helpful assistant that examines if a {node_type} satisfies a given query and assign a score from 0.0 to 1.0. If the {node_type} does not satisfy the query, the score should be 0.0. If there exists explicit and strong evidence supporting that {node_type} satisfies the query, the score should be 1.0. If partial evidence or weak evidence exists, the score should be between 0.0 and 1.0.\n"
                        f"Please score the {node_type} based on how well it satisfies the query. ONLY output the floating point score WITHOUT anything else. "
                        f"Here is the query:\n\"{query}\"\n" +
                        f"Here is the information about the {node_type}:\n" +
                        # f"{self.kb.get_doc_info(node_id, add_rel=True)}" +
                        f"{' '.join(self.kb.get_doc_info(node_id, add_rel=True, compact=True).split()[:1000])}." +
                        f"Output: The numeric score of this {node_type} is: " 
                    )
    
            success = False
            for _ in range(self.max_cnt):    
                # try:
                answer = get_llm_output(prompt, 
                                        self.llm_model_pt,
                                        self.tokenizer,
                                        self.llm_model_name
                                        )
                print(f'====================answer is {answer}===================')
                answer = find_floating_number(answer)
                if len(answer) >= 1:
                    answer = answer[0]
                    success = True
                    break

            if success:
                llm_score = float(answer)
                sim_score = (cand_len - idx) / cand_len
                score = llm_score + self.sim_weight * sim_score
                pred_dict[node_id] = score
            else:
                return initial_score_dict, initial_score_dict
        
        return pred_dict, initial_score_dict
        
    