import argparse

def ArgumentParser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_size', type=int, default="11", help="num_size")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--window', type=int, default="10", help="window_size")
    parser.add_argument('--stride', type=int, default="5", help="stride")
    parser.add_argument('--batch_size', type=int, default="128", help="batch_size")
    parser.add_argument('--hidden_size', type=int, default="128", help="hidden_size")
    parser.add_argument('--code_size', type=int, default="64", help="code_size")
    parser.add_argument('--num_epoch', type=int, default="500", help="num_epoch")
    parser.add_argument('--query_num', type=int, default="5", help="query_num")
    parser.add_argument('--lr', type=float, default="2e-4", help="learning rate")
    parser.add_argument('--la', type=float, default="10", help="la")
    parser.add_argument('--lambda_1', type=float, default="0.01", help="lambda_1")
    parser.add_argument('--lambda_2', type=float, default="0.01", help="lambda_2")
    parser.add_argument('--dataset', type=str,
                        default='air_quality', help='The dataset to use')

    return parser.parse_args()