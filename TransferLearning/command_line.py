"""
Command Line Parser.
"""
import argparse
import runner


if __name__ == "__main__":
    # Create the Parser
    parser = argparse.ArgumentParser()

    # Add Argument
    parser.add_argument('--content', type=str, required=True, nargs=2)
    parser.add_argument('--style', type=str, required=True,nargs=2)
    parser.add_argument('-v', '--verbose', type=bool, required=False)

    # Parse the argument
    args = parser.parse_args()
    print(args)

    # Pass arguments to Functions
    runner.Run(args.content, args.style, args.verbose)
