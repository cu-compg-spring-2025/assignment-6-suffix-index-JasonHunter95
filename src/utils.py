import random
import numpy as np
import gzip
import os
import subprocess


def read_fasta(file):
    if file.endswith('.gz'):
        with gzip.open(file, 'rt') as f:
            data = f.read().split('>')
            data = [x for x in data if x != '']
            data = [x.split('\n') for x in data]
            data = [[x[0], ''.join(x[1:]).upper()] for x in data]
        return data
    else:
        with open(file, 'r') as f:
            data = f.read().split('>')
            data = [x for x in data if x != '']
            data = [x.split('\n') for x in data]
            data = [[x[0], ''.join(x[1:]).upper()] for x in data]
        return data

def get_kmers(seq, k):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def sim_reads(seq, read_length, num_reads, error_rate):
    kmers = get_kmers(seq, read_length)
    seed_reads = [random.choice(kmers) for i in range(num_reads)]

    reads = [] 

    for read in seed_reads:
        error_mask = np.random.poisson(error_rate, read_length)
        flips = np.where(error_mask == 1)[0]
        read_array = list(read)
        for flip in flips:
            new_char = random.choice('ACGT')
            read_array[flip] = new_char
        read = ''.join(read_array)
        reads.append(read)

    return reads

def visualize_region(full_sequence, region_str, max_size, reference_path, output_prefix=None, use_tree=False):
    """
    Create a suffix trie/tree visualization for a specific region of a sequence
    
    Parameters:
        full_sequence (str): The complete sequence
        region_str (str): Region in format 'start-end'
        max_size (int): Maximum allowed region size
        reference_name (str): Name of the reference for output filename
        output_prefix (str): Directory prefix for output files
        use_tree (bool): Use tree instead of trie
        
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
        
        # build appropriate data structure based on use_tree flag
        if use_tree:
            # Import build_suffix_tree dynamically to avoid circular imports
            from suffix_tree import build_suffix_tree
            data_structure = build_suffix_tree(region_sequence)
            structure_type = "tree"
        else:
            # Import build_suffix_trie dynamically to avoid circular imports
            from suffix_trie import build_suffix_trie
            data_structure = build_suffix_trie(region_sequence)
            structure_type = "trie"
        
        reference_filename = os.path.basename(reference_path)
        reference_name = os.path.splitext(reference_filename)[0]
        
        # remove the .fa.gz extension
        if '.' in reference_name:
            reference_name = reference_name.split('.')[0]
        
        # visualize the region
        region_name = f"{reference_name}_{start}-{end}"
        dot_output = to_dot(data_structure)
        
        # set up directories based on output_prefix
        if output_prefix:
            dot_dir = os.path.join('dots', output_prefix)
            png_dir = os.path.join('pngs', output_prefix)
        else:
            dot_dir = os.path.join('dots', f'suffix_{structure_type}')
            png_dir = os.path.join('pngs', f'suffix_{structure_type}')
        
        # ensure directories exist
        os.makedirs('dots', exist_ok=True)
        os.makedirs('pngs', exist_ok=True)
        
        dot_file = os.path.join(dot_dir, f'{region_name}.dot')        
        
        with open(dot_file, 'w') as f:
            f.write(dot_output)
            
        generate_png_from_dot(dot_file, png_dir)
        return True, f"Generated visualization for region {start}-{end}"
        
    except ValueError:
        return False, "Error: Region should be in format 'start-end', e.g., '100-300'"
    
    
def to_dot(structure):
    """ Return dot representation of suffix tree to make a picture """
    lines = []
    
    # Check if we're dealing with a list-based suffix tree
    if isinstance(structure, list) and len(structure) > 0 and isinstance(structure[0], list):
        lines.append('digraph "Suffix tree" {')
        lines.append('  node [shape=circle];')
        
        # Process all nodes
        for i, node in enumerate(structure):
            # Add node with its substring as label if it's not empty
            label = node[0] if node[0] else "ROOT"
            lines.append(f'  {i} [label="{label}"];')
            
            # Add edges to children
            for char, child_idx in node[1].items():
                lines.append(f'  {i} -> {child_idx} [ label="{char}" ];')
        
        lines.append('}')
    else:
        # Original code for dictionary-based tries
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
        _to_dot_helper(structure, 0)
        lines.append('}')
        
    return '\n'.join(lines) + '\n'

def generate_png_from_dot(dot_file, png_dir='png'):
    """Convert a DOT file to PNG using the dot command"""
    # grab the filename
    filename = os.path.basename(dot_file)
    # grab the file path for the images
    if png_dir is None:
        png_dir = os.path.join('.', 'pngs')

    os.makedirs(png_dir, exist_ok=True)

    # create the name of each image
    png_file = os.path.join(png_dir, filename.replace('.dot', '.png'))
    
    try: # run the console command but in a function instead of a shell script
        subprocess.run(['dot', '-Tpng', dot_file, '-o', png_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating PNG: {e}")

