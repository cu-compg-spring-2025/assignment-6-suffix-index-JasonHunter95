import argparse
import utils
import psutil
import os
import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='Suffix Trie')

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
    
    # new arguments for region selection so I can make tries with genomic data from the FASTA file
    parser.add_argument('--region',
                        help='Region to visualize (format: start-end)',
                        type=str)
    
    parser.add_argument('--max-size',
                        help='Maximum sequence length to visualize (default: 200)',
                        type=int,
                        default=100)

    return parser.parse_args()

def build_suffix_trie(s, show_progress=False):
    s += '$'
    root = {}
    iterator = tqdm.tqdm(range(len(s)), desc="Building suffix trie") if show_progress else range(len(s))
    for i in iterator:
        current = root
        for char in s[i:]:
            if char not in current:
                current[char] = {}
            current = current[char]
    return root

def search_trie(trie, pattern):
    current = trie ## start at the root
    match_len = 0 
    
    for i, char in enumerate(pattern):
        if char in current: # each node contains a dictionary of children thats nested horribly in the console output
            current = current[char]
            match_len += 1
        else:
            break
    return match_len

def main():
    args = get_args()
    
    T = None
    
    # create script-specific directories
    dot_dir = os.path.join('dots', 'suffix_trie')
    png_dir = os.path.join('pngs', 'suffix_trie')
    
    # create directories if they don't exist
    os.makedirs(dot_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    # passing in the string directly for testing
    if args.string:
        T = args.string
        trie = build_suffix_trie(T)
                
        dot_output = utils.to_dot(trie)
        dot_file = f'{args.string}.dot'
        dot_path = os.path.join(dot_dir, dot_file)
        
        with open(dot_path, 'w') as f:
            f.write(dot_output)
            
        utils.generate_png_from_dot(dot_path, png_dir)
        if args.query:
            for query in args.query:
                match_len = search_trie(trie, query)
                print(f'{query} : {match_len}')
                
    # passing in the FASTA file
    elif args.reference:
        reference = utils.read_fasta(args.reference)
        T = reference[0][1]
        
        # just handle a specific region
        if args.region:
            success, message = utils.visualize_region(T, args.region, args.max_size, args.reference, output_prefix='suffix_trie', use_tree=False)
            print(message)
        else:
            print("No region specified. Skipping visualization for large sequence.")
            
        
        # if args.query:
        #     print(f"Warning: Building suffix trie for complete sequence ({len(T)} bp)")
        #     print(f"This may consume significant memory and time.")
            # only build the trie if we have queries to search for so other things work properly
            # literally needs to be sent to a supercomputer to run this!!!
            # response = input("Continue? (y/n): ")
            # if response.lower() != 'y':
            #     print("Exiting.")
            #     return
            # else:
            #     print(f"Attempting to build suffix trie for complete sequence ({len(T)} bp) hopefully your computer is FAT")
            #     trie = build_suffix_trie(T)
            #     print("Trie built, processing queries...")
            #     for query in args.query:
            #         match_len = search_trie(trie, query, T)
            #         print(f'{query} : {match_len}')

if __name__ == '__main__':
    main()