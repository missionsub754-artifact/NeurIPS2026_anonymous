import argparse

# Parameter setting
def args_parser():
    parser = argparse.ArgumentParser(description='X-CoSD')
    parser.add_argument(
        "--draft-model", type=str, required=True, help="Draft model path."
    )
    parser.add_argument(
        "--target-model", type=str, required=True, help="Target model path."
    )
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path.")
    parser.add_argument("--fp16", action="store_true", help="use float16 dtype.")

    parser.add_argument("--tot-iter", type=int, default=1, 
        help="Number of total iterations",
    )

    parser.add_argument("--n-config", type=int, default="4", 
        help="Number of candidate tokens per round",
    )

    parser.add_argument("-K", type=int, default=20, 
        help="Number of replacement candidates with SR-DV method",
    )
    
    parser.add_argument("-M", type=int, default=10, 
        help="Number of iteration rounds in SR-DV method",
    )

    parser.add_argument("--u-s", type=int, default=20, 
        help="Number of temp. perturbation in U-HLM",
    )

    parser.add_argument("--u-max", type=float, default=2.0, 
        help="Max. theta in U-HLM",
    )

    parser.add_argument("--u-th", type=float, default=0.5, 
        help="Uncertainty threshold in U-HLM",
    )

    parser.add_argument(
        "--datapath", type=str, required=True, help="The json data file."
    )

    parser.add_argument("--max-new-tokens", type=int, default=128)

    parser.add_argument(
        "--verification-method", type=str, default="hr", choices=["hr", "ul", "dl", "gr","tr", "srdv", "uhlm"],
        help="HR/UL/DL(Naive)/GR/TR/SR-DV/U-HLM methods."
    )

    parser.add_argument("--disable-tqdm", action="store_true")

    parser.add_argument("--run-baseline", action="store_true")

    args = parser.parse_args()
    return args