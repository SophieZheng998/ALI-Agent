import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', type=int, default=1,
                        help='Specify which gpu to use.')

    parser.add_argument('--seed', type=int, default=101,
                        help='Random seed.')
    
    parser.add_argument('--temperature', type = float, default=1.0)

    parser.add_argument('--llm_name', type=str, default= 'gpt-4-1106-preview',
                        help="Specify which llm to test. ['gpt-4-1106-preview', 'gpt-3.5-turbo-1106', 'gemini-pro', 'vicuna-7b', 'vicuna-13b', 'vicuna-33b, 'llama2-7b', 'llama2-13b', 'llama2-70b', 'chatglm3-6b']")

    parser.add_argument("--type", type = str, default = 'stereotype',
                        help = "[stereotype, ethic, law]")

    parser.add_argument("--dataset", type = str, default = 'stereotype_decodingtrust',
                        help = "[stereotype_decodingtrust, stereotype_crows, ethic_ETHICS, ethic_socialchemistry, law_traffic, law_harmfulbehavior]")

    parser.add_argument('--saveID', type=str, default= '',
                        help='The name of the saving log.')

    parser.add_argument("--train", action="store_true")
    
    parser.add_argument("--baseline", action="store_true",
                        help="evaluation on baselines")

    # intention + web browsing
    parser.add_argument("--web_browsing", action="store_true")

    parser.add_argument("--start_from", type=int, default = 0,
                        help="specify which regulation to start from")

    # harmful behavior + gptfuzz
    parser.add_argument("--plug_in_jb", action='store_true', help='')

    parser.add_argument('--model_type', type=str, choices=['api', 'local'],
                        help="Specify model type. Options are 'api' or 'local'.")

    parser.add_argument('--api_key', type=str, default='',
                        help='API key for accessing the specified model API.')

    parser.add_argument('--api_endpoint', type=str, default='',
                        help='The API endpoint to use if model_type is "api".')

    parser.add_argument('--local_model_path', type=str, default='',
                        help='The local model path to use if model_type is "local".')
    
    parser.add_argument('--gpt_base_url', type=str, default='',
                            help='Base URL for the GPT API.')

    parser.add_argument('--gpt_model_name', type=str, default='',
                            help='Name of the GPT model to use.')

    parser.add_argument('--gpt_api_key', type=str, default='',
                            help='API key for accessing the GPT model.')
    
    parser.add_argument('--bing_api_key', type=str, default='',
                        help='API key for accessing Bing services.')
    parser.add_argument('--bing_gpt_key', type=str, default='',
                        help='gpt_key for bing.')
    
    parser.add_argument('--bing_gpt_name', type=str, default='',
                        help='gpt name for accessing Bing services.')
    
    parser.add_argument('--customer_config_id', type=str, default='')
    
    parser.add_argument('--bing_base_url', type=str, default='')


    args, _ = parser.parse_known_args()

    return args


