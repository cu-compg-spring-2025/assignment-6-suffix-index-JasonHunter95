import argparse
import utils
import psutil
import subprocess
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

def build_suffix_trie(s, show_progress=True):
    s += '$'
    root = {}
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1024 / 1024
    available_mem = psutil.virtual_memory().available / 1024 / 1024
    print("Total available memory: {:.2f} MB".format(available_mem))
    
    iterator = tqdm.tqdm(range(len(s)), desc="Building suffix trie") if show_progress else range(len(s))
    # for memory reporting purposes
    report_interval = max(1, len(s) // 10)  # reports ~10 times

    for i in iterator:
        current = root
        for char in s[i:]:
            if char not in current:
                current[char] = {}
            current = current[char]
            
        # report memory usage
        if i % report_interval == 0:
            current_mem = process.memory_info().rss / 1024 / 1024
            mem_increase = current_mem - start_mem
            print(f"Memory usage: {current_mem:.2f} MB (+{mem_increase:.2f} MB)")
    return root

def search_trie(trie, pattern, s):
    current = trie ## start at the root
    match_len = 0 
    
    for i, char in enumerate(pattern):
        if char in current: # each node contains a dictionary of children thats nested horribly in the console output
            # so we it
            # move to the next node down
            current = current[char]
            match_len += 1
            # print(f"Matched '{char}' at position {i}, current match length: {match_len}")
        else:
            # print(f"Did not match '{char}', breaking the loop here at {i}")
            return 0
    
    if(match_len - 1 == len(s)):
        # print(f"Matched the entirety of the deepest node of the trie: {s}")
        return match_len
    else:
        # print(f"Matched the pattern: {pattern}")
        return match_len
    
def visualize_region(full_sequence, region_str, max_size, reference_path):
    """
    Create a suffix trie visualization for a specific region of a sequence
    
    Parameters:
        full_sequence (str): The complete sequence
        region_str (str): Region in format 'start-end'
        max_size (int): Maximum allowed region size
        reference_name (str): Name of the reference for output filename
        
    Returns:
        tuple: (success (bool), message (str))
    """
    try:
        start, end = map(int, region_str.split('-'))
        
        # validate region
        if start < 0 or end > len(full_sequence) or start >= end:
            return False, f"Invalid region: {start}-{end}. Sequence length is {len(full_sequence)}."
        
        # check size limit
        if end - start > max_size:
            print(f"Warning: Selected region exceeds max size ({max_size}). Truncating.")
            end = start + max_size
            
        # extract region and build trie
        region_sequence = full_sequence[start:end]
        trie_region = build_suffix_trie(region_sequence)
        
        reference_filename = os.path.basename(reference_path)
        reference_name = os.path.splitext(reference_filename)[0]
        
        # remove the .fa.gz extension
        if '.' in reference_name:
            reference_name = reference_name.split('.')[0]
        
        # visualize the region
        region_name = f"{reference_name}_{start}-{end}"
        dot_output = to_dot(trie_region)
        
        # ensure directories exist
        os.makedirs('dots', exist_ok=True)
        os.makedirs('pngs', exist_ok=True)
        
        dot_file = os.path.join('dots', f'{region_name}.dot')
        
        
        with open(dot_file, 'w') as f:
            f.write(dot_output)
            
        generate_png_from_dot(dot_file)
        return True, f"Generated visualization for region {start}-{end}"
        
    except ValueError:
        return False, "Error: Region should be in format 'start-end', e.g., '100-300'"


def to_dot(trie):
    """ Return dot representation of trie to make a picture """
    lines = []
    node_counter = [0]  # use a list to allow modification in nested function
    
    def _to_dot_helper(node, parid):
        current_id = node_counter[0]
        node_counter[0] += 1
        
        for char, child in node.items():
            child_id = node_counter[0]
            lines.append(f'  {current_id} -> {child_id} [ label="{char}" ];')
            _to_dot_helper(child, child_id)
        
    lines.append('digraph "Suffix trie" {')
    lines.append('  node [shape=circle label=""];')
    _to_dot_helper(trie, 0)
    lines.append('}')
    return '\n'.join(lines) + '\n'

def generate_png_from_dot(dot_file):
    """Convert a DOT file to PNG using the dot command"""
    # grab the filename
    filename = os.path.basename(dot_file)
    # grab the file path for the images
    png_dir = os.path.join('.', 'pngs')

    # create the name of each image
    png_file = os.path.join(png_dir, filename.replace('.dot', '.png'))
    
    try: # run the console command but in a function instead of a shell script
        subprocess.run(['dot', '-Tpng', dot_file, '-o', png_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating PNG: {e}")


    

def main():
    args = get_args()
    


    T = None

    # passing in the string directly for testing
    if args.string:
        T = args.string
        trie = build_suffix_trie(T)
                
        dot_output = to_dot(trie)
        dot_file = f'{args.string}.dot'
        dot_directory = os.path.join('dots', dot_file)
        
        with open(dot_directory, 'w') as f:
            f.write(dot_output)
            
        generate_png_from_dot(dot_directory)
        if args.query:
            for query in args.query:
                match_len = search_trie(trie, query, T)
                print(f'{query} : {match_len}')
                
    # passing in the FASTA file
    elif args.reference:
        reference = utils.read_fasta(args.reference)
        T = reference[0][1]
        
        # just handle a specific region
        if args.region:
            success, message = visualize_region(T, args.region, args.max_size, args.reference)
            print(message)
        else:
            print("No region specified. Skipping visualization for large sequence.")
            
        
        # if args.query:
        #     # only build the trie if we have queries to search for so other things work properly
        #     # literally needs to be sent to a supercomputer to run this!!!
        #     print(f"Attempting to build suffix trie for complete sequence ({len(T)} bp)...")
        #     trie = build_suffix_trie(T)
        #     print("Trie built, processing queries...")
        #     for query in args.query:
        #         match_len = search_trie(trie, query, T)
        #         print(f'{query} : {match_len}')



if __name__ == '__main__':
    main()
