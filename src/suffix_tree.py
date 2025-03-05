import argparse
import utils
import os

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


def add_suffix(nodes, suf):
    n = 0
    i = 0
    while i < len(suf):
        b = suf[i] 
        children = nodes[n][CHILDREN]
        if b not in children:
            n2 = len(nodes)
            nodes.append([suf[i:], {}])
            nodes[n][CHILDREN][b] = n2
            return
        else:
            n2 = children[b]

        sub2 = nodes[n2][SUB]
        j = 0
        while j < len(sub2) and i + j < len(suf) and suf[i + j] == sub2[j]:
            j += 1

        if j < len(sub2):
            n3 = n2 
            n2 = len(nodes)
            nodes.append([sub2[:j], {sub2[j]: n3}])
            nodes[n3][SUB] = sub2[j:]
            nodes[n][CHILDREN][b] = n2

        i += j
        n = n2

def build_suffix_tree(text):
    text += "$"

    nodes = [ ['', {}] ]

    for i in range(len(text)):
        add_suffix(nodes, text[i:])
    
    return nodes

def search_tree(suffix_tree, P):
    n = 0
    i = 0
    while i < len(P):
        b = P[i]
        children = suffix_tree[n][CHILDREN]
        if b not in children:
            return i
        else:
            n = children[b]

        sub = suffix_tree[n][SUB]
        j = 0
        while j < len(sub) and i + j < len(P) and P[i + j] == sub[j]:
            j += 1

        if j < len(sub):
            return i + j

        i += j

    return len(P)

def main():
    args = get_args()

    T = None
    
    # create script-specific directories
    dot_dir = os.path.join('dots', 'suffix_tree')
    png_dir = os.path.join('pngs', 'suffix_tree')
    
    # create directories if they don't exist
    os.makedirs(dot_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    if args.string:
        T = args.string
        tree = build_suffix_tree(T)
        
        dot_output = utils.to_dot(tree)
        dot_file = f'{args.string}.dot'
        dot_path = os.path.join(dot_dir, dot_file)
        
        with open(dot_path, 'w') as f:
            f.write(dot_output)
            
        utils.generate_png_from_dot(dot_path, png_dir)
        
        if args.query:
            for query in args.query:
                match_len = search_tree(tree, query)
                print(f'{query} : {match_len}')
        
        
    # passing in the FASTA file
    elif args.reference:
        reference = utils.read_fasta(args.reference)
        T = reference[0][1]
        
        # just handle a specific region
        if args.region:
            success, message = utils.visualize_region(T, args.region, args.max_size, args.reference, output_prefix='suffix_tree', use_tree=True)
            print(message)
        else:
            print("No region specified. Skipping visualization for large sequence.")
    


if __name__ == '__main__':
    main()
