import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', type=int, default=2,
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

    args, _ = parser.parse_known_args()

    return args


