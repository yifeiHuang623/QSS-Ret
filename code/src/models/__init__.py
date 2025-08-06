from .vss import  VSS
from .llm_reranker import LLMReranker
from .multi_vss import MultiVSS
from .qss import QSS

def get_model(args, kb, llm_model_pt=None, tokenizer=None):
    model_name = args.model
    if model_name == 'VSS':
        return VSS(
            kb,
            emb_model=args.emb_model,
            query_emb_dir=args.query_emb_dir, 
            candidates_emb_dir=args.node_emb_dir
        )
    if model_name == "QSS":
        return QSS(
            kb,
            emb_model=args.emb_model,
            query_emb_dir=args.query_emb_dir, 
            candidates_emb_dir=args.node_emb_dir,
            dataset=args.dataset,
            llm_model=llm_model_pt,
            tokenizer=tokenizer,
        )
        
    if model_name == 'MultiVSS':
        return MultiVSS(
            kb,
            emb_model=args.emb_model,
            query_emb_dir=args.query_emb_dir, 
            candidates_emb_dir=args.node_emb_dir,
            chunk_emb_dir=args.chunk_emb_dir,
            aggregate=args.aggregate,
            chunk_size=args.chunk_size,
            max_k=args.multi_vss_topk
        )
    if model_name == 'LLMReranker':
        return LLMReranker(kb, 
                           emb_model=args.emb_model,
                           llm_model_pt=llm_model_pt,
                           tokenizer=tokenizer,
                           llm_model=args.llm_model,
                           query_emb_dir=args.query_emb_dir, 
                           candidates_emb_dir=args.node_emb_dir,
                           max_cnt = args.max_retry,
                           max_k=args.llm_topk,
                           chunk_emb_dir=args.chunk_emb_dir
                           )

    raise NotImplementedError(f'{model_name} not implemented')