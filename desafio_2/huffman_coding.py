import re
import numpy as np
from PIL import Image
import os

print("Huffman Compression Program")
print("=================================================================")

def file_to_str(file_path):
    string = np.asarray(Image.open(file_path),np.uint8)
    shape = string.shape
    save_string = string
    string = str(string.tolist())
    return save_string, string, shape

def get_letters(string):
    letters = []
    only_letters = []
    for letter in string:
        if letter not in letters:
            frequency = string.count(letter)
            letters.append(frequency)
            letters.append(letter)
            only_letters.append(letter)
    return frequency, letters, only_letters

def get_tree(letters):
    nodes = []
    while len(letters) > 0:
        nodes.append(letters[0:2])
        letters = letters[2:]
    nodes.sort()
    huffman_tree = []
    huffman_tree.append(nodes)
    return nodes, huffman_tree

def combine_nodes(nodes, huffman_tree):
    pos = 0
    newnode = []
    if len(nodes) > 1:
        nodes.sort()
        nodes[pos].append("1")
        nodes[pos+1].append("0")
        combined_node1 = (nodes[pos] [0] + nodes[pos+1] [0])
        combined_node2 = (nodes[pos] [1] + nodes[pos+1] [1])
        newnode.append(combined_node1)
        newnode.append(combined_node2)
        newnodes=[]
        newnodes.append(newnode)
        newnodes = newnodes + nodes[2:]
        nodes = newnodes
        huffman_tree.append(nodes)
        combine_nodes(nodes, huffman_tree)

    return nodes, huffman_tree

def get_checklist(nodes, huffman_tree):
    newnodes, huffman_tree = combine_nodes(nodes, huffman_tree)

    huffman_tree.sort(reverse = True)

    checklist = []
    for level in huffman_tree:
        for node in level:
            if node not in checklist:
                checklist.append(node)
            else:
                level.remove(node)
    count = 0
    for level in huffman_tree:
        count+=1
    print()

    return checklist

def get_binary(only_letters, string, checklist):
    letter_binary = []
    if len(only_letters) == 1:
        letter_code = [only_letters[0], "0"]
        letter_binary.append(letter_code*len(string))
    else:
        for letter in only_letters:
            code =""
            for node in checklist:
                if len (node)>2 and letter in node[1]:           #genrating binary code
                    code = code + node[2]
            lettercode =[letter,code]
            letter_binary.append(lettercode)

    bitstring =""
    for character in string:
        for item in letter_binary:
            if character in item:
                bitstring = bitstring + item[1]
    binary ="0b"+bitstring
    return letter_binary, binary, bitstring

def compare_n_write(string,binary,bitstring,filename):
    uncompressed_file_size = len(string)*7
    compressed_file_size = len(binary)-2
    print("Your original file size was", uncompressed_file_size,"bits. The compressed size is:",compressed_file_size)
    print("This is a saving of ",uncompressed_file_size-compressed_file_size,"bits")
    output = open("./result/"+filename+".txt","w+")
    output = open("./result/"+filename+".txt","w+")
    output.write(bitstring)

def undo(binary,letter_binary,shape,save_string,filename):
    bitstring = str(binary[2:])
    uncompressed_string =""
    code =""
    for digit in bitstring:
        code = code+digit
        pos=0                                        #iterating and decoding
        for letter in letter_binary:
            if code ==letter[1]:
                uncompressed_string=uncompressed_string+letter_binary[pos] [0]
                code=""
            pos+=1

    temp = re.findall(r'\d+', uncompressed_string)
    res = list(map(int, temp))
    res = np.array(res)
    res = res.astype(np.uint8)
    res = np.reshape(res, shape)
    print("Observe the shapes and input and output arrays are matching or not")
    print("Input image dimensions:",shape)
    print("Output image dimensions:",res.shape)
    data = Image.fromarray(res)
    data.save('./result/'+filename+'.png')
    if save_string.all() == res.all():
        print("Success")


def run(filename):
    pathname = os.path.realpath(__file__)
    pathname = os.path.split(pathname)[0]
    path_database_imgs = os.path.join(pathname, 'imgs', '')
    
    save_string, string, shape = file_to_str(path_database_imgs+filename)
    frequency, letters, only_letters = get_letters(string)
    nodes, huffman_tree = get_tree(letters)
    nodes, huffman_tree = combine_nodes(nodes, huffman_tree)
    checklist = get_checklist(nodes, huffman_tree)
    letter_binary, binary, bitstring = get_binary(only_letters,string,checklist)
    compare_n_write(string,binary,bitstring,'result_'+filename)
    undo(binary,letter_binary,shape,save_string,'result_'+filename)

filename = 'Compressao_1.png'
run(filename)
filename = 'Compressao_2.png'
run(filename)