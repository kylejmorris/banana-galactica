# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

import galai as gal

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    model = gal.load_model("standard")

if __name__ == "__main__":
    download_model()
