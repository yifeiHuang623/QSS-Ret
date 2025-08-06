import os.path as osp
import json
import os
import argparse
from tqdm import tqdm
from src.benchmarks import get_qa_dataset, get_semistructured_data
from openai import OpenAI
import logging
import time

client = OpenAI(
            base_url = "xxx",
            api_key = "xxx"
        )

def generate_response(message):
    template = '''
        Suppose you are a question understanding expert. Now you need to understand a question and extract the important information according to the graph meta data.

        This is a product network graph. The graph meta data includes node type and edge type.
        Node type:
            {'product', 'brand'}
        Edge type:
            [('product', 'also_buy', 'product'),
            ('product', 'also_view', 'product'),
            ('product', 'has_brand', 'brand')]

        You need to output a query graph according to the question and graph meta data. Please think step by step.
        First, you can extract the important information from the question and combine the information from the same item, make sure there is no left.
        Second, you need to output a query graph according to the graph meta data and the extracted important information.
        Output the nodes and edges in a json format whose keys are 'nodes' and 'edges'. 
        The value of 'nodes' is a list, each item represent a node and includes (node id, node type and node information from the question). 
        The value of 'edges' is a list, each item represent an edge and includes two node id (node id 1, node id 2).

        For example, the question is 'Looking for a vibrant red inflatable float that deflates for easy storage and pairs well with the Rob Allen 12 Liter OVERBLOWN Foam Spearfishing Float and Optional Dive Flag from Adidas. The product also need to be cheap and easy-use. Any recommendations?'
        First, you need to extract important information from the question and combine the information from the same item, 'a vibrant red inflatable float that deflates for easy storage' and 'cheap and easy-use' seems to describe the same product, 'pairs well with the Rob Allen 12 Liter OVERBLOWN Foam Spearfishing Float and Optional Dive Flag from Adidas' seems to describe another product.
        You can induce that the question includes a product which is 'a vibrant red inflatable float that deflates for easy storage, cheap and easy-use' and has edges with another product that is 'Rob Allen 12 Liter OVERBLOWN Foam Spearfishing Float and Optional Dive Flag' and its brand is 'Adidas'.
        So following the graph meta information, the graph includes nodes and edges: {'nodes': [(1, 'product', 'a vibrant red inflatable float that deflates for easy storage and cheap and easy-use'), (2, 'product', 'Rob Allen 12 Liter OVERBLOWN Foam Spearfishing Float and Optional Dive Flag'),(3, 'brand', 'adidas')], 'edges':[(1,2),(2,3)]}
        Please put the answer node at the first place.

        Following the above example, the question is ''' + message + '''
        Only output the nodes and edges in a json format as the example and nothing else.
    '''
    
    template_2 = '''
        Suppose you are a question understanding expert. Now you need to understand a question and extract the important information according to the graph meta data.

        This is a academic network graph. The graph meta data includes node type and edge type.
        Node type:
            {'paper', 'author', 'institution', 'field_of_study'}
        Edge type:
            ['author___affiliated_with___institution',
            'paper___cites___paper',
            'paper___has_topic___field_of_study',
            'author___writes___paper']

        You need to output a query graph according to the question and graph meta data. Please think step by step.
        First, you can extract the important information from the question and combine the information from the same item, make sure there is no left.
        Second, you need to output a query graph according to the graph meta data and the extracted important information.
        Output the nodes and edges in a json format whose keys are 'nodes' and 'edges'. 
        The value of 'nodes' is a list, each item represent a node and includes (node id, node type and node information from the question). 
        The value of 'edges' is a list, each item represent an edge and includes two node id (node id 1, node id 2).
        
        For example, the question is "Are there any papers about optical from King’s College London? ".
        To answer the question, you need to induce what node type does King’s College London belong, it may be an institution.
        Then follow the node type and find the edge type author___affiliated_with___institution, so you can get authors from King’s College London. The information of the author is None.
        From the node type author, you can find the edge type author___writes___paper, so you can get papers from King’s College London.
        To get optical papers, you need induce from the paper to find contents about optical.
        So following the graph meta information, the graph includes nodes and edges: {'nodes':[(1, 'paper', 'about optical'), (2, 'author', 'None'), (3, 'institution', 'King's College London')], edges: [(1, 2), (2, 3)]}
        Please put the answer node at the first place.

        Following the above example, the question is ''' + message + '''
        Only output the nodes and edges in a json format as the example and nothing else.
    '''

    completion = client.chat.completions.create(
        model="meta/llama3-70b-instruct",
        messages=[{"role":"user","content":template_2}],
        temperature=0.1,
        top_p=1,
        max_tokens=1024
        )

    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response = chunk.choices[0].delta.content
            print(response)
    
    return response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mag", choices=['amazon', 'primekg', 'mag'])

    parser.add_argument("--split", default="test")

    # can eval on a subset only
    parser.add_argument("--test_ratio", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # create file handler that logs debug and higher level messages
    os.makedirs(f"./query/logs/", exist_ok=True)
    fh = logging.FileHandler(f"./query/logs/{str(time.time())}.log")
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)


    kb = get_semistructured_data(args.dataset)
    qa_dataset = get_qa_dataset(args.dataset)

    split_idx = qa_dataset.get_idx_split(test_ratio=args.test_ratio)

    indices = split_idx[args.split].tolist()

    output_json = []

    for idx in tqdm(indices):
        query, query_id, answer_ids, meta_info = qa_dataset[idx]
        response = generate_response(query)
    
        output_dict = {}
        output_tmp = {}
        output_tmp["response"] = response
        output_tmp["id"] = idx
        output_tmp["query"] = query
        
        output_dict[query_id] = output_tmp
        
        output_json.append(output_dict)
        logger.info(f"query_id: {query_id}, response: {response}")
        with open(f'./query/graph_structure_{args.dataset}.json', 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=4)
        time.sleep(20)
        

        