import os.path as osp
import torch    
from typing import Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import re
import json
from src.models.vss import VSS

import os
import sys

from src.benchmarks import get_qa_dataset, get_semistructured_data
from src.models.model import ModelForSemiStructQA
import copy

from src.tools.api_lib.llama import complete_text_llama, complete_text_qwen

def find_floating_number(text):
    pattern = r'0\.\d+|1\.0|0|1'
    matches = re.findall(pattern, text)
    return [round(float(match), 4) for match in matches if float(match) <= 1.1]

def find_answer(text):
    yes_index = text.lower().find("yes")
    no_index = text.lower().find("no")
    if yes_index != -1:
        return [1]

    return [0]


def handle_query_graph(query_graph: str, model:str='gpt-4o'):
        def extract_json(query_graph: str):
            front_num = 0
            front = -1
            back = -1
            for i in range(len(query_graph)):
                if query_graph[i] == '{': 
                    front_num += 1
                    if front_num == 1:
                        front = i
                elif query_graph[i] == '}': 
                    front_num -= 1
                    if front_num == 0:
                        back = i
                        break
            if back == -1:
                return {}
        
            query_str = query_graph[front:back+1].replace('\n', '').replace('(','[').replace(')',']')
            print(query_str)
            return json.loads(query_str)
        
        def extract_json_llama(query_graph: str):
            front_num = 0
            front = -1
            back = -1
            for i in range(len(query_graph)):
                if query_graph[i] == '{': 
                    front_num += 1
                    if front_num == 1:
                        front = i
                elif query_graph[i] == '}': 
                    front_num -= 1
                    if front_num == 0:
                        back = i
                        break
            if back == -1:
                return {}
            
            query_str = query_graph[front:back+1].replace('\\\'','\'').replace('\'','\"').replace('(','[').replace(')',']').replace('\n','')
            
            query_str = re.sub(r"(?<=[A-Za-z])\"(?=[A-Za-z])",'\'', query_str)
            return json.loads(query_str)
        

        if model == "llama":
            query_json = extract_json_llama(query_graph)
        else:
            query_json = extract_json(query_graph)
        nodes = query_json["nodes"]
        edges = query_json["edges"]
        nodes_new = [[]]
        nodes_ids = []
        edges_new = []
        for node in nodes:
            nodes_new.append(node)
            nodes_ids.append(node[0])

        for edge in edges:
            if edge[0] not in nodes_ids or edge[1] not in nodes_ids:
                continue
            if edge not in edges_new:
                edges_new.append(edge)
            if [edge[1], edge[0]] not in edges_new:
                edges_new.append([edge[1], edge[0]])

        if len(edges_new) == 0:
            for i in range(1, len(nodes_new)):
                for j in range(i+1, len(nodes_new)):
                    edges_new.append((nodes_new[i][0], nodes_new[j][0]))
                    edges_new.append((nodes_new[j][0], nodes_new[i][0]))
                    
        neighbors = {}
        for edge in edges_new:
            if edge[0] in neighbors.keys():
                neighbors[int(edge[0])].append(int(edge[1]))
            else:
                neighbors[int(edge[0])] = [int(edge[1])]

        return nodes_new, edges_new, neighbors

def get_top_k(score_dict, ids, top_K = 50):
        node_ids = list(ids)
        node_scores = list(score_dict)

        # get the ids with top k highest scores
        top_k_idx = torch.topk(torch.FloatTensor(node_scores),
                               min(top_K, len(node_scores)),
                               dim=-1).indices.view(-1).tolist()
        top_k_node_ids = [node_ids[i] for i in top_k_idx]
        return top_k_node_ids

class QSS(ModelForSemiStructQA):
    
    def __init__(self, 
                 kb,
                 query_emb_dir, 
                 candidates_emb_dir,
                 tokenizer, 
                 llm_model,
                 max_cnt=3,
                 aggregate='top3_avg',
                 emb_model='text-embedding-ada-002',
                 max_k=50,
                 dataset="mag"):
        '''
        Vector Similarity Search
        Args:
            kb (src.benchmarks.semistruct.SemiStruct): kb
            query_emb_dir (str): directory to query embeddings
            candidates_emb_dir (str): directory to candidate embeddings
        '''
        
        super(QSS, self).__init__(kb)
        # self.emb_model = emb_model
        self.emb_model = SentenceTransformer("../../llm_model/gte-large-en-v1.5", trust_remote_code=True)
        
        self.reranker = "llama"
        
        self.generate_model = "gpt-4o"
        
        if self.generate_model == "llama":
            query_name = f"../../query/graph_structure_total_{dataset}_llama.json"
        else:
            query_name = f"../../query/graph_structure_final_{dataset}.json"
            
        with open(query_name) as f:
            self.query_response = json.load(f)
        # self.query_emb_dir = query_emb_dir
        # self.candidates_emb_dir = candidates_emb_dir
        self.total_ids = list(range(kb.num_nodes()))

        self.chunk = torch.load(f'../..//emb/{dataset}/gte/doc/test_emb_chunk_dict.pt')

        self.short = torch.load(f"../..//emb/{dataset}/gte/doc/short_emb_chunk_dict.pt")
        
        candidate_emb_dict = torch.load(f'../..//emb/{dataset}/gte/doc/candidate_emb_dict.pt')
        candidate_embs = [candidate_emb_dict[idx] for idx in self.candidate_ids]
        self.candidate_embs = torch.cat(candidate_embs, dim=0)
        
        self.max_k = max_k
        self.aggregate = aggregate
        
        self.parent_vss = VSS(kb, query_emb_dir, candidates_emb_dir, emb_model=emb_model)
        
        self.dataset = dataset
        
        self.max_cnt = max_cnt
        self.llm_model_pt = llm_model
        self.tokenizer = tokenizer
        
        self.types = "reranker"
        
        self.without_motif = False
        
        
    def bfs(self, nodes, node_id, top_k_node_ids_list, neighbors, text_count:int=5, node_count:int=3):
        queue = [node_id]
        queue_graph = [1]
        visit = set()
        visit_graph = set()
        
        candidate_subgraph = []
        
        if self.dataset == "amazon":
            text_count = 3
            node_count = 1

        # 深度优先搜索
        if len(neighbors) == 0:
            return []
        while len(queue) != 0:
            item = queue.pop()
            item_graph = queue_graph.pop()
            visit.add(item)
            visit_graph.add(item_graph)
            neighbors_kb = self.kb.get_neighbor_nodes(item)
            for neighbor in neighbors[item_graph]:
                if neighbor in visit_graph: continue
                candidate_node = top_k_node_ids_list[neighbor][:text_count]
                
                if nodes[neighbor][1] != "paper" and nodes[neighbor][1] != "product":
                    right_node = []
                    for node in candidate_node:
                        if len(right_node) >= node_count: break
                        if self.kb.get_node_type_by_id(node) == nodes[neighbor][1]:
                            right_node.append(node)
                    candidate_node = right_node
                    
                if nodes[neighbor][2] != "None":  
                    right_neighbors = set(neighbors_kb) & set(candidate_node)
                    for n in right_neighbors:
                        node_type = self.kb.get_node_type_by_id(n)
                        if self.dataset == "amazon":
                            node_text = self.kb[n].brand_name if node_type != "product" else self.kb[n].title
                        else:
                            node_text = self.kb[n].DisplayName if node_type != "paper" else self.kb[n].title
                        candidate_subgraph.append((node_type, node_text))
                else:
                    right_neighbors = set(neighbors_kb)
                    
                if len(right_neighbors) != 0:
                    queue.extend(right_neighbors - visit)
                    queue_graph.extend(len(right_neighbors - visit) * [neighbor])
        
            if (len(visit) ==  len(nodes) - 1):
                return candidate_subgraph
            
        # 并返回邻居文本
            
        return []
    
    
    def build_template(self, node_id, query, subgraph):
            
        neighbor_info = {}
        for node in subgraph:
            if node[0] not in neighbor_info.keys(): 
                neighbor_info[node[0]] = [node[1]]
            else:
                neighbor_info[node[0]].append(node[1])
                
        if self.dataset == "amazon":
            node_type = "product"
        else:
            node_type = "paper"
            
        original_template = (f"You are a helpful assistant that examines if a {node_type} satisfies a given query and assign a score from 0.0 to 1.0. If the {node_type} does not satisfy the query, the score should be 0.0. If there exists explicit and strong evidence supporting that {node_type} satisfies the query, the score should be 1.0. If partial evidence or weak evidence exists, the score should be between 0.0 and 1.0.\n"
                            f"Please score the {node_type} based on how well it satisfies the query. ONLY output the floating point score WITHOUT anything else. "
                            f"Here is the query:\n\"{query}\"\n" +
                            f"Here is the information about the {node_type}:\n" +
                            f"{' '.join(self.kb.get_doc_info(node_id, add_rel=True, compact=True).split()[:1000])}" +
                            f"Output: The numeric score of this {node_type} is: ")
                
        if len(neighbor_info) == 0 or self.without_motif:
            if self.types == "reranker":
                return (f"You are a helpful assistant that examines if a {node_type} satisfies a given query and assign a score from 0.0 to 1.0. If the {node_type} does not satisfy the query, the score should be 0.0. If there exists explicit and strong evidence supporting that {node_type} satisfies the query, the score should be 1.0. If partial evidence or weak evidence exists, the score should be between 0.0 and 1.0.\n"
                            f"Please score the {node_type} based on how well it satisfies the query. ONLY output the floating point score WITHOUT anything else. "
                            f"Here is the query:\n\"{query}\"\n" +
                            f"Here is the information about the {node_type}:\n" +
                            f"{' '.join(self.kb.get_doc_info(node_id, add_rel=True, compact=True).split()[:1000])}" +
                            f"Output: The numeric score of this {node_type} is: "), original_template
            else:
                return (f"You are a helpful assistant that examines if a {node_type} satisfies a given query. Please pay attention to the details of this {node_type} and make a cautious decision. " +
                        f"Output YES or NO first and then output a brief explanation to support the answer." +
                f"Here is the query:\n\"{query}\"\n" +
                f"Here is the information about the {node_type}:\n" 
                f"{' '.join(self.kb.get_doc_info(node_id, add_rel=True, compact=True).split()[:1000])}")
        
        if self.dataset == "amazon":
            structure_text = ""
            if "brand" in neighbor_info.keys():
                structure_text += f"This {node_type}'s brand is {neighbor_info['brand']}."
            if "product" in neighbor_info.keys():
                structure_text += f"People who bought this {node_type} also bought/viewed the {neighbor_info['product']}" 
            
            if structure_text != "":
                structure_text += f"\n Below is the detailed information of this {node_type}:\n"
            
        else:
            structure_text = ""
            if "author" in neighbor_info.keys():
                structure_text += f"This {node_type}'s author is {neighbor_info['author']}."
            if "institution" in neighbor_info.keys():
                structure_text += f"This {node_type} comes from {neighbor_info['institution']}"
            if "field_of_study" in neighbor_info.keys():
                structure_text += f"The field of study of this {node_type} is {neighbor_info['field_of_study']}"
            if "paper" in neighbor_info.keys():
                # structure_text += f"This paper cites/is cited by {neighbor_info['paper']}"
                structure_text += f"This paper has coauthors with {neighbor_info['paper']}"
            
            if structure_text != "":
                structure_text += f"\n Below is the detailed information of this {node_type}:\n"
            
        if self.types == "reranker":
            template = (f"You are a helpful assistant that examines if a {node_type} satisfies a given query and assign a score from 0.0 to 1.0. If the {node_type} does not satisfy the query, the score should be 0.0. If there exists explicit and strong evidence supporting that {node_type} satisfies the query, the score should be 1.0. If partial evidence or weak evidence exists, the score should be between 0.0 and 1.0.\n" +
            f"Please score the {node_type} based on how well it satisfies the query. ONLY output the floating point score WITHOUT anything else. " +
            f"Here is the query:\n\"{query}\"\n" +
            f"Here is the information about the {node_type}:\n" +
            f"{structure_text}." +
            f"{' '.join(self.kb.get_doc_info(node_id, add_rel=False, compact=True).split()[:1000])}." +
            f"Output: The numeric score of this {node_type} is: " )
        else:
            template = (f"You are a helpful assistant that examines if a {node_type} satisfies a given query. Please output YES or NO first and then output a brief explanation"
            f"Here is the query:\n\"{query}\"\n" +
            f"Here is the information about the {node_type}:\n" +
            f"{structure_text}." +
            f"{' '.join(self.kb.get_doc_info(node_id, add_rel=False, compact=True).split()[:1000])}")
            
        return template, original_template


    def forward(self, 
                query: str,
                query_id: int,
                **kwargs: Any):
        query_graph = self.query_response[str(query_id)]["response"]
        nodes, edges, neighbors = handle_query_graph(query_graph, self.generate_model)
        # queries = [] + [query] + [node_info[2] for node_info in nodes[1:]]
        queries = [[], query] + [node_info[2] for node_info in nodes[2:] if len(nodes) > 1]
        
        initial_score_dict, _, _ = self.parent_vss(query, query_id)
        node_ids = list(initial_score_dict.keys())
        node_scores = list(initial_score_dict.values())
        top_k_idx = torch.topk(torch.FloatTensor(node_scores),
                               min(self.max_k, len(node_scores)),
                               dim=-1
                               ).indices.view(-1).tolist()

        top_k_node_ids = [node_ids[i] for i in top_k_idx]
        pred_top_k = copy.deepcopy(initial_score_dict)
        
        top_k_node_ids_list = [[]]
        queries_emb = torch.from_numpy(self.emb_model.encode(queries))
        
        if self.dataset == "amazon":
            multi_vss = True
        else:
            multi_vss = False
            
        # 按照multi-vss再重排一下
        pred_top_k_tmp = {}
        if multi_vss:
            for node_id in top_k_node_ids:
                similarity = torch.matmul(queries_emb[1].cuda(), self.chunk[str(node_id)].cuda().T).cpu().view(-1)
                pred_top_k[node_id] = torch.max(similarity).item()
                pred_top_k_tmp[node_id] = pred_top_k[node_id]
                
            top_k_idx = torch.topk(torch.FloatTensor(list(pred_top_k_tmp.values())),
                                   min(self.max_k, len(pred_top_k_tmp)),
                                   dim=-1).indices.view(-1).tolist()
            top_k_node_ids = [list(pred_top_k_tmp.keys())[i] for i in top_k_idx]
            
        for idx in range(queries_emb.shape[0]):
            if idx == 0: continue
            if idx == 1:
                top_k_node_ids_list.append(top_k_node_ids)
                continue
            query_emb = queries_emb[idx]
            if nodes[idx][1] == "paper" or nodes[idx][1] == "product":  
                similarity = torch.matmul(query_emb.cuda(), 
                                  self.candidate_embs.view(len(self.candidate_ids), -1).cuda().T).cpu().view(-1)      
                top_k_node_ids_list.append(get_top_k(similarity, self.candidate_ids, top_K=50))
            else:
                sim = torch.matmul(query_emb.cuda(), 
                                torch.cat(list(self.short.values()), dim=0).cuda().T).cpu().view(-1)
                top_k_node_ids_list.append(get_top_k(sim, self.short.keys()))
              
        add_node = []  
        candidate_subgraph = {}
        for node_id in tqdm(top_k_node_ids):
            subgraph = self.bfs(nodes, node_id, top_k_node_ids_list, neighbors)
            if len(subgraph) != 0:
                add_node.append(node_id)
                pred_top_k[node_id] += 1000
                candidate_subgraph[node_id] = subgraph
                
        pred_top_k_original = copy.deepcopy(pred_top_k)
        
        # reranker
        candidate_node = top_k_node_ids[:20] if len(candidate_subgraph) == 0 else add_node[:20]
        
        text_len = 0
        origin_text_len = 0
        for node_id in candidate_node:
            text, origin_text = self.build_template(node_id, query, subgraph=(candidate_subgraph[node_id] if node_id in candidate_subgraph.keys() else []))
        
            success = False
            for _ in range(self.max_cnt):    
                if self.reranker == "qwen":
                    answer = complete_text_qwen(message=text, 
                                        llm_model_pt=self.llm_model_pt,
                                        tokenizer=self.tokenizer)
                else:
                    answer = complete_text_llama(message=text, 
                                        llm_model_pt=self.llm_model_pt,
                                        tokenizer=self.tokenizer)
                
                print(f'====================answer is {answer}===================')
                if self.types == "reranker":
                    answer = find_floating_number(answer)
                else:
                    answer = find_answer(answer)
                if len(answer) >= 1:
                    answer = answer[0]
                    success = True
                    break

            if success:
                llm_score = float(answer)
                score = llm_score * 1000
                pred_top_k[node_id] += score
            else:
                return pred_top_k, len(add_node), pred_top_k_original
            
            text_len += len(text.split())
            origin_text_len += len(origin_text.split())
            
        return pred_top_k, len(add_node), pred_top_k_original,  initial_score_dict, (1.0 * text_len / len(candidate_node)), (1.0 * origin_text_len / len(candidate_node))