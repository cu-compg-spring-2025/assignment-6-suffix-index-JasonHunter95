import argparse
import os
import sys

# trying a few different import approaches based on how the script is being run
try:
    # first try relative import first (for use when imported as module)
    from . import ukkonens_suffix_tree
    from . import utils
except ImportError:
    try:
        # try package import (for tests)
        import src.ukkonens_suffix_tree as ukkonens_suffix_tree
        import src.utils as utils
    except ImportError:
        # try direct import (for direct script execution)
        import ukkonens_suffix_tree
        import utils
        
def get_args():
    parser = argparse.ArgumentParser(description='Suffix Tree to Suffix Array Converter')
    parser.add_argument('--reference',
                        help='Reference sequence file',
                        type=str)
    parser.add_argument('--string',
                        help='Reference sequence',
                        type=str)
    parser.add_argument('--query',
                        help='Query sequences',
                        nargs='+',
                        type=str)
    parser.add_argument('--region',
                        help='Region to visualize (format: start-end)',
                        type=str)
    parser.add_argument('--max-size',
                        help='Maximum sequence length to visualize (default: 200)',
                        type=int,
                        default=100)
    return parser.parse_args()

def build_suffix_array(T):
    """
    Build a suffix array from the input string T using a suffix tree constructed
    via Ukkonen's algorithm.
    
    This method builds the suffix tree and then performs a DFS (visiting children
    in lexicographic order) to extract leaf nodes’ suffix indices.
    """
    suffix_tree = ukkonens_suffix_tree.SuffixTree(T)
    suffix_array = []

    def dfs(node):
        # If the node is a leaf, append its stored suffix index.
        if not node.children:
            suffix_array.append(node.index)
        else:
            # Traverse children in lexicographic order (by edge-starting character).
            for key in sorted(node.children.keys()):
                dfs(node.children[key])
                
    dfs(suffix_tree.root)
    return suffix_array

def search_array(T, suffix_array, q):
    def prefix_overlap(s, q): 
        i = 0
        while i < len(s) and i < len(q) and s[i] == q[i]:
            i += 1
        return i

    lo = -1
    hi = len(suffix_array)
    
    while (hi - lo > 1):
        mid = (lo + hi) // 2
        if T[suffix_array[mid]:] < q:
            lo = mid
        else:
            hi = mid
    result = prefix_overlap(T[suffix_array[hi]:], q)
    return result

def main():
    args = get_args()
    T = None

    txt_dir = os.path.join('txts', 'ukkonens_suffix_array')
    os.makedirs(txt_dir, exist_ok=True)

    if args.string:
        T = args.string
        array = build_suffix_array(T)
        
        txt_file = f'{args.string}_ukkonens_suffix_array.txt'
        txt_path = os.path.join(txt_dir, txt_file)
        utils.visualize_suffix_array(T, array, txt_path)
        
        if args.query:
            for query in args.query:
                match_len = search_array(T, array, query)
                print(f'{query} : {match_len}')
                
    elif args.reference:
        reference = utils.read_fasta(args.reference)
        T = reference[0][1]
        
        if args.region:
            try:
                start, end = map(int, args.region.split('-'))
                
                if start < 0 or end > len(T) or start >= end:
                    print(f"Invalid region: {start}-{end}. Sequence length is {len(T)}.")
                    return
                
                if end - start > args.max_size:
                    print(f"Warning: Selected region exceeds max size ({args.max_size}). Truncating.")
                    end = start + args.max_size
                    
                region = T[start:end]
                array = build_suffix_array(region)
                
                reference_name = os.path.basename(args.reference).split('.')[0]
                txt_file = f'{reference_name}_{start}-{end}_ukkonens_suffix_array.txt'
                txt_path = os.path.join(txt_dir, txt_file)
                
                utils.visualize_suffix_array(region, array, txt_path)
                print(f"Generated Ukkonen's suffix array visualization for region {start}-{end}")
                
                if args.query:
                    for query in args.query:
                        match_len = search_array(region, array, query)
                        print(f'{query} : {match_len}')
            except ValueError:
                print("Error: Region should be in format 'start-end', e.g., '100-300'")
        else:
            print("No region specified. Skipping visualization for large sequence.")

if __name__ == '__main__':
    main()