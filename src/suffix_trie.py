import argparse
import utils

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
    trie = [ ['', {}] ]  ## [ prefix (str), {char: idx} ]
    
    for i in range(len(s)):
        current = 0
        for j in range(i, len(s)):
            char = s[j]
            children = trie[current][1]
            
            if char not in children:
                new_idx = len(trie)
                trie.append([s[j:], {}])
                children[char] = new_idx
                break
            else:
                next_idx = children[char]
                substring = trie[next_idx][0]
                k = 0
                while k < len(substring) and j + k < len(s) and s[j + k] == substring[k]:
                    k += 1            
                if k < len(substring):
                    old_idx = next_idx
                    new_idx = len(trie)
                    trie.append([substring[:k], {substring[k]: old_idx}])
                    trie[old_idx][0] = substring[k:]
                    children[char] = new_idx
                    
                current = next_idx
    return trie


def search_trie(trie, pattern):
    current = 0
    matched = 0
    i = 0
    
    # for debugging/testing purposes
    # print(f"Searching for: '{pattern}'")
    # print(f"Trie: {trie}")
    
    while i < len(pattern):
        children = trie[current][1]
        char = pattern[i]
        
        # print(f"  Position {i}, char '{char}', current node {current}")
        # print(f"  Available children: {children}")
        
        if char in children:
            next_node = children[char]
            node_str = trie[next_node][0]
            
            # print(f"  Found char '{char}', moving to node {next_node} with string '{node_str}'")

            
            # match as many characters as possible from this node
            j = 0
            while i < len(pattern) and j < len(node_str) and pattern[i] == node_str[j]:
                print(f"    Matching {pattern[i]} == {node_str[j]}")
                matched += 1
                i += 1
                j += 1
                
            current = next_node
            
            # if it didn't match the full node string, gtfo
            if j < len(node_str):
                # print(f"  Didn't match full node string, stopping at {matched} matches")
                break
        else:
            # print(f"  No child for '{char}', stopping at {matched} matches")
            break
            
    # print(f"Final match count: {matched}")
    return matched
    

def main():
    args = get_args()

    T = None

    if args.string:
        T = args.string
    elif args.reference:
        reference = utils.read_fasta(args.reference)
        T = reference[0][1]

    trie = build_suffix_trie(T)
    
    # print(trie)

    if args.query:
        for query in args.query:
            match_len = search_trie(trie, query)
            print(f'{query} : {match_len}')

if __name__ == '__main__':
    main()
