import argparse


def main():
    parser = argparse.ArgumentParser(description="Credit-card fraud detection with a chronological evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="Compare models on the chronological validation split")
    threshold = commands.add_parser("threshold", help="Freeze cost-minimising thresholds on validation")
    threshold.add_argument("--review-costs", type=float, nargs="+", default=[1., 5., 10., 25.])
    threshold.add_argument("--headline-cost", type=float, default=5.)
    final = commands.add_parser("final", help="Score the test period once with the frozen model and thresholds")
    final.add_argument("--policy", default="outputs/threshold")
    for sub, folder in [(evaluate, "evaluate"), (threshold, "threshold"), (final, "final")]:
        sub.add_argument("--data", default="data/creditcard.csv")
        sub.add_argument("--output", default="outputs/" + folder)
        sub.add_argument("--n-bootstrap", type=int, default=1000)
        sub.add_argument("--seed", type=int, default=42)
    for sub in [evaluate, threshold]:
        sub.add_argument("--train-fraction", type=float, default=.6)
        sub.add_argument("--val-fraction", type=float, default=.2)
        sub.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.command == "evaluate":
        from .evaluate import run
    elif args.command == "threshold":
        from .threshold import run
    else:
        from .final import run
    try:
        run(args)
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
