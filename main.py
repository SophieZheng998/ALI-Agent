from parse import parse_args
from simulation.utils import fix_seeds
from simulation.arena import Arena
import os
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_config_from_file(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)
    



if __name__ == '__main__':

    args = parse_args()
    fix_seeds(args.seed) 

    arena_ = Arena(args)

    if args.web_browsing:
        _,laws = arena_.execute_browsing(args.bing_api_key,args.bing_gpt_name,args.bing_gpt_key,args.customer_config_id,args.bing_base_url,gradio=False,input = None)
        arena_.Testing_law(laws)
    elif args.baseline and not args.train:
        if args.type == 'stereotype':
            arena_.execute_stereotype_baselines()
        elif args.type == 'ethic':
            arena_.execute_ethic_baselines()
        elif args.type == 'law':
            arena_.execute_law_baselines()
        else:
            print("Not implemented")
    else:
        if args.type == 'stereotype':
            arena_.execute_stereotype()
        elif args.type == 'ethic':
            arena_.execute_ethic()
        elif args.dataset == 'law_traffic':
            arena_.execute_law()
        elif args.dataset == 'law_harmfulbehavior':
            arena_.execute_multi_round_harm()
        else:
            print("Not implemented")