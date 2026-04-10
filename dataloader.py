"""
This script downloads the Tiny Shakespeare dataset from URL
and saves it locally as a text file named "shakespeare.txt".

You can change the save location by modifying the DATA_PATH variable.
"""
import requests
import os

url = "https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt"
DATA_PATH = "shakespeare.txt"
def download_dataset()->None:
    if os.path.exists(DATA_PATH):
        print("File already exits. No changes made.")
        return

    text_file = requests.get(url).text
    with open(DATA_PATH,"w") as f:
        f.write(text_file)


def load_dataset(print_text = False)->str:
    """
    Loads the dataset from DATA_PATH into a string.

    Args:
        print_text (bool): If True, prints the first 500 characters
                           of the dataset for preview.

    Returns:
        str: The full dataset content as a single string.
    """
    with open(DATA_PATH, "r") as f:
        txt = f.read()

    print("Total Characters in text: ", len(txt))
    if print_text == True:
        print(txt[:1000]) 

    return txt

if __name__ == "__main__":
    download_dataset()
    load_dataset(print_text=True)



    
