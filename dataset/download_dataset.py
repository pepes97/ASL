import os
import zipfile
import gdown

'''
Download and extract ASL dataset
'''


def download_dataset():
    URL = "https://drive.google.com/uc?export=download&id=1KS5PakuPwOnL6zRardIaxpQ44uAjRwk_"
    gdown.download(URL, os.getcwd()+os.sep+"ASL_Alphabet.zip", quiet=False)
    extract_dataset()
    print("\n")

def extract_dataset():
  if os.path.exists(os.getcwd()+os.sep+"ASL_Alphabet.zip"):
    print("Extracting the archive")
    with zipfile.ZipFile(os.getcwd()+os.sep+"ASL_Alphabet.zip", 'r') as zip_ref:
      zip_ref.extractall(os.getcwd()+os.sep)
    
    print("Done")
    os.remove(os.getcwd()+os.sep+"ASL_Alphabet.zip")  
    print("Zips removed")      

if __name__ == '__main__':

    download_dataset()