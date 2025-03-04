import argparse
import utils
import subprocess
import os

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

    return parser.parse_args()

def build_suffix_trie(s):
    s += '$'
    root = {}
    for i in range(len(s)):
        current = root
        for char in s[i:]:
            if char not in current:
                current[char] = {}
            current = current[char]
    return root


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
        print(f"Generated PNG image: {png_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating PNG: {e}")


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
            print(f"Did not match '{char}', breaking the loop here at {i}")
            return 0
    
    if(match_len - 1 == len(s)):
        print(f"Matched the entirety of the deepest node of the trie: {s}")
        return match_len
    else:
        print(f"Matched the pattern: {pattern}")
        return match_len
    

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
        trie = build_suffix_trie(T)
        
        print(trie)
        
        dot_output = to_dot(trie)
        dot_file = f'{args.reference}.dot'
        
        # this might be too crazy for the reference file so I'll probably comment it out
        with open(dot_file, 'w') as f:
            f.write(dot_output)
        generate_png_from_dot(dot_file)
            
        
        if args.query:
            for query in args.query:
                match_len = search_trie(trie, query, T)
                print(f'{query} : {match_len}')



if __name__ == '__main__':
    main()
