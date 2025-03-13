import argparse
import os
import sys

# trying a few different import approaches based on how the script is being run
try:
    # first try relative import first (for use when imported as module)
    from . import suffix_tree
    from . import utils
except ImportError:
    try:
        # try package import (for tests)
        import src.suffix_tree as suffix_tree
        import src.utils as utils
    except ImportError:
        # try direct import (for direct script execution)
        import suffix_tree
        import utils

SUB = 0
CHILDREN = 1

def get_args():
    parser = argparse.ArgumentParser(description='Suffix Tree')

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
    tree = suffix_tree.build_suffix_tree(T)
    suffixes = []

    # BFS traversal of suffix tree
    stack = [(0, "")]
    
    # print("Starting BFS Traversal...\n")
    while stack:
        node_idx, current_suffix = stack.pop()
        node = tree[node_idx]
        substring, children = node[0], node[1]
        
        # if we are at a leaf node (no children left)
        if not children:
            full_suffix = (current_suffix + substring).rstrip('$')
            if not full_suffix:
                continue
            suffix_position = len(T) - len(full_suffix)
            suffixes.append(suffix_position)
        
        # add children to stack
        for char, child_idx in children.items():
            stack.append((child_idx, current_suffix + substring))
            
    suffixes = sorted(suffixes, key=lambda x: T[x:])  # sort based on the actual suffix strings

    return suffixes

def search_array(T, suffix_array, q):

    # checks how many characters at the beginning of the string s and query q are the same
    # looping through the string and query until the characters are no longer the same or the end of either the string/query is reached
    def prefix_overlap(s, q): 
        
        i = 0
        while i < len(s) and i < len(q) and s[i] == q[i]:
            i += 1
        return i

    # all indices in the suffix array will be covered with these two pointers
    lo = -1
    hi = len(suffix_array)
    
    while (hi - lo > 1):
        mid = int((lo + hi) / 2)
        if T[suffix_array[mid]:] < q:
            lo = mid
        else:
            hi = mid
    result = prefix_overlap(T[suffix_array[hi]:], q)
    return result

def main():
    args = get_args()

    T = None

    # create script-specific directories
    txt_dir = os.path.join('txts', 'suffix_array')
    
    # create directories if they don't exist
    os.makedirs(txt_dir, exist_ok=True)

    if args.string:
        T = args.string
        array = build_suffix_array(T)
        
        # create and save visualization
        txt_file = f'{args.string}_suffix_array.txt'
        txt_path = os.path.join(txt_dir, txt_file)
        utils.visualize_suffix_array(T, array, txt_path)
        
        if args.query:
            for query in args.query:
                match_len = search_array(T, array, query)
                print(f'{query} : {match_len}')
                
    # passing in the FASTA file
    elif args.reference:
        reference = utils.read_fasta(args.reference)
        T = reference[0][1]
        
        # just handle a specific region
        if args.region:
            try:
                start, end = map(int, args.region.split('-'))
                
                # validate region
                if start < 0 or end > len(T) or start >= end:
                    print(f"Invalid region: {start}-{end}. Sequence length is {len(T)}.")
                    return
                
                # check size limit
                if end - start > args.max_size:
                    print(f"Warning: Selected region exceeds max size ({args.max_size}). Truncating.")
                    end = start + args.max_size
                    
                region = T[start:end]
                array = build_suffix_array(region)
                
                # create visualization
                reference_name = os.path.basename(args.reference).split('.')[0]
                txt_file = f'{reference_name}_{start}-{end}_suffix_array.txt'
                txt_path = os.path.join(txt_dir, txt_file)
                
                utils.visualize_suffix_array(region, array, txt_path)
                print(f"Generated suffix array visualization for region {start}-{end}")
                
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